"""
Probe (not in run_all): what the "standing still" pauses of the one-leg stage look like with a model -
corners touching, capture-point margin, pelvis speed, how fast the motors move (and the quiet_motors_term score) - to check the still_stable metric is fair.
    python tests/probe_still.py [model.pt] [seconds] [--zero] [--stand]
    --zero: reference motion only, no policy; --stand: the stand stage (still from 1 s in, no pushes)
    --switch: one-leg stage, but in the pauses the network is told it is in the stand stage (two skills of one
              network: stand still in the pauses, lift in between) - does it stand quieter, and are the
              changes over smooth (falls)?
"""
import math
import sys

import numpy as np
import torch

from _common import make_world

from bereshit.addons.PPO import Academy, Trainer
import robot as R
import motions as M
import WalkAgent as W
from walk_metrics import capture_point, support_margin

args = [a for a in sys.argv[1:] if not a.startswith("--")]
Academy.setup_trainer(W.make_config(None))
if "--noise" in sys.argv:        # sample the actions like in training (exploration noise), not the best one
    plain_act = Trainer.act
    Trainer.act = lambda self, observation, deterministic=False, **kwargs: plain_act(self, observation, deterministic=False, **kwargs)
if "--zero" in sys.argv:
    def zero_act(self, observation, deterministic=False, **kwargs):
        return np.zeros(W.ACTION_DIM, dtype=np.float32), None, {
            "value": torch.tensor(0.0), "log_prob": torch.tensor(0.0), "policy_version": 0,
            "observation": torch.zeros(W.OBS_DIM), "continuous_action": torch.zeros(W.ACTION_DIM)}
    Trainer.act = zero_act
    Academy.get_trainer().inference_only = True
else:
    Academy.load_trained_model(args[0] if args else "model.pt")
seconds = float(args[1]) if len(args) > 1 else 8.0

root, robot = R.build_robot()
floor = R.build_floor()
STAND = "--stand" in sys.argv
agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage="stand" if STAND else "one_leg", pushes=False)
root.add_component(agent)
world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
agent.first_leg = 0
if "--switch" in sys.argv:
    TASK_SLICE = slice(-(3 + len(M.TASKS)), -3)      # sense(): ..., task one-hot, command speed, sin/cos yaw
    plain_sense = agent.sense

    def switched_sense(contacts, reference):
        o = plain_sense(contacts, reference)
        if M.one_leg_still(agent.motion_time):
            o[TASK_SLICE] = [1.0 if name == "stand" else 0.0 for name in M.TASKS]
        return o
    agent.sense = switched_sense

ticks = round(W.CONTROL_DT / W.TICK)


def measured(t):
    return t > 1.0 if STAND else M.one_leg_still(t, W.ONE_LEG_SETTLE)


def shown(t):
    return STAND or M.one_leg_still(t)


# Every control step inside the measured part of the pauses (from W.ONE_LEG_SETTLE s in), like the pass test:
# which of its conditions fail, and which way the pelvis moves (forward / sideways), vs the whole body (CoM).
fails = {"a foot off the floor": 0, "capture point outside the feet": 0, "pelvis >= 5 cm/s": 0}
ok = n = 0
sums = {"pelvis forward": 0.0, "pelvis sideways": 0.0, "CoM forward": 0.0, "CoM sideways": 0.0,
        "pelvis pitch rate": 0.0, "pelvis roll rate": 0.0}
motor_speed = quiet = 0.0
per_pause = {}               # pause number in the episode -> [still moments, moments, motor deg/s sum]
# The steady streak (W.standing_well): per pause the longest run of "standing well" moments, and how often
# each of its conditions fails.
streak = {}                  # pause number -> [current run s, longest run s]
runs = []                    # length (s) of every finished run of standing well
steady_fails = {"a corner up": 0, "capture point out": 0, "pelvis speed": 0, "tilt": 0, "turning": 0}
previous_angles = np.array([s.angle() for s in agent.servos])
print("  t    pause  corners L/R  icp margin  pelvis fwd / side cm/s  CoM fwd / side cm/s  pitch / roll rate deg/s")
for step in range(int(seconds / W.TICK)):
    world.update(step % ticks == 0)
    if step % ticks:
        continue
    t = agent.motion_time                # the motion's clock (it waits in the one-leg pauses)
    corners = agent._foot_corners()
    down = [c[1] < 0.003 for c in corners]
    touching = [(c[0], c[2]) for c, d in zip(corners, down) if d]
    com, com_v = agent._center_of_mass(), agent._com_velocity()
    icp = capture_point((com[0], com[2]), (com_v[0], com_v[2]), com[1])
    margin = support_margin(icp, touching)
    v = agent.pelvis.Rigidbody.velocity
    w = R.world_to_local(agent.pelvis.transform.quaternion, agent.pelvis.Rigidbody.angular_velocity)         if hasattr(R, "world_to_local") else agent.pelvis.Rigidbody.angular_velocity
    speed = math.hypot(v.x, v.z)
    angles = np.array([s.angle() for s in agent.servos])
    motors = np.abs(angles - previous_angles) / W.CONTROL_DT
    previous_angles = angles
    row = (v.x * 100, v.z * 100, com_v[0] * 100, com_v[2] * 100, math.degrees(w.z), math.degrees(w.x))
    if measured(t):
        n += 1
        both = any(down[:4]) and any(down[4:])
        fails["a foot off the floor"] += not both
        fails["capture point outside the feet"] += margin <= 0.0
        fails["pelvis >= 5 cm/s"] += speed >= 0.05
        ok += both and margin > 0.0 and speed < 0.05
        p = per_pause.setdefault(0 if STAND else int(t // (M.ONE_LEG_CYCLE / 2)), [0, 0, 0.0])
        p[0] += both and margin > 0.0 and speed < 0.05
        p[1] += 1
        p[2] += float(motors.mean())
        av = agent.pelvis.Rigidbody.angular_velocity
        tilt, yaw = math.hypot(av.x, av.z), av.y
        run = streak.setdefault(0 if STAND else int(t // (M.ONE_LEG_CYCLE / 2)), [0.0, 0.0])
        well = W.standing_well([float(d) for d in down], margin, speed, tilt, yaw)
        if not well and run[0] > 0.0:
            runs.append(run[0])
        run[0] = run[0] + W.CONTROL_DT if well else 0.0
        run[1] = max(run[1], run[0])
        for k, bad in zip(steady_fails, (not all(down), margin <= 0.0, speed >= W.STEADY_SPEED,
                                          math.degrees(tilt) >= W.STEADY_TILT_DEG,
                                          math.degrees(abs(yaw)) >= W.STEADY_YAW_DEG)):
            steady_fails[k] += bad
        motor_speed += float(motors.mean())
        quiet += W.quiet_motors_term(motors)
        for k, x in zip(sums, row):
            sums[k] += abs(x)
    if step % (5 * ticks) == 0 and shown(t):
        print(f"{t:5.2f}  {'still' if measured(t) else 'settl'}  {sum(down[:4])}/{sum(down[4:])}"
              f"          {margin * 100:+6.1f} cm   {row[0]:+6.1f} / {row[1]:+6.1f}          {row[2]:+6.1f} / {row[3]:+6.1f}"
              f"          {row[4]:+6.1f} / {row[5]:+6.1f}")
if n:
    print(f"measured moments: {n}, standing still {ok / n:.0%}; failing: " +
          ", ".join(f"{k} {v / n:.0%}" for k, v in fails.items()))
    print("average size: " + ", ".join(f"{k} {v / n:.1f}" for k, v in sums.items()) + " (cm/s, deg/s)")
    print(f"falls: {agent.falls}, episode time {agent.episode_time:.1f} s for {agent.motion_time:.1f} s of motion "
          f"(pauses wait for {agent.curriculum.steady_hold:.2f} s of standing well in a row)")
    if not STAND:
        print("per pause (0 = the episode start, the others come right after a weight-back):")
        for k, (a, b, m) in sorted(per_pause.items()):
            print(f"  pause {k} at {k * M.ONE_LEG_CYCLE / 2:5.1f} s: still {a / b:4.0%}, motors {m / b:5.1f} deg/s, "
                  f"longest steady streak {streak[k][1]:.2f} s")
    runs += [r[0] for r in streak.values() if r[0] > 0.0]      # runs still going at the end
    print(f"standing well: {sum(runs) / (n * W.CONTROL_DT):.0%} of the measured moments, in {len(runs)} runs; "
          f"runs of >= 0.1 s: {sum(r >= 0.1 - 1e-9 for r in runs)}, >= 0.5 s: {sum(r >= 0.5 - 1e-9 for r in runs)}, "
          f">= 1 s: {sum(r >= 1.0 - 1e-9 for r in runs)}, longest {max(runs, default=0.0):.2f} s")
    print("standing well (steady streak) fails on: " + ", ".join(f"{k} {v / n:.0%}" for k, v in steady_fails.items()))
    print(f"motors: average speed {motor_speed / n:.1f} deg/s, quiet_motors_term {quiet / n:+.2f}")
