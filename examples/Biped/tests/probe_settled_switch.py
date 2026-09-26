"""
Probe (not in run_all): two skills of one network in the one-leg stage. After each weight-back the one-leg skill
keeps control until the robot has found its balance again - close to the standing start the stand skill knows -
and only then is the network told it is in the stand stage, until the next weight shift. The time to find the
balance is not limited: the one-leg clock waits at the end of the pause until it is reached (at most WAIT_MAX s,
reported as "never").
Per pause: how long finding the balance took, then from that moment to the end of the pause how still it stands
(the still_stable test) and how fast the motors move.
    python tests/probe_settled_switch.py [model.pt] [seconds] [--plain] [--noise]
    --plain: never switch (the same balance wait and measurement, for comparison)
    --noise: sample the actions like in training (exploration noise) instead of the best action
"""
import math
import sys

import numpy as np

from _common import make_world

from bereshit.addons.PPO import Academy, Trainer
import robot as R
import motions as M
import WalkAgent as W
from walk_metrics import capture_point, support_margin

# "Balance found": both feet flat, pelvis slower than the still_stable limit, hardly tilting, the center of
# mass back over its start spot - held for BALANCE_HOLD s.
BALANCE_SPEED = 0.05       # m/s
BALANCE_TILT = 10.0        # deg/s, pitch and roll
BALANCE_COM = 0.02         # m from the start spot
BALANCE_HOLD = 0.1         # s
WAIT_MAX = 10.0            # s: give up waiting and report "never"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PLAIN, NOISE = "--plain" in sys.argv, "--noise" in sys.argv
Academy.setup_trainer(W.make_config(None))
Academy.load_trained_model(args[0] if args else "model.pt")
if NOISE:
    plain_act = Trainer.act
    Trainer.act = lambda self, observation, deterministic=False, **kwargs: plain_act(self, observation, deterministic=False, **kwargs)
seconds = float(args[1]) if len(args) > 1 else 60.0

root, robot = R.build_robot()
floor = R.build_floor()
agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage="one_leg", pushes=False)
root.add_component(agent)
world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
agent.first_leg = 0

HALF = M.ONE_LEG_CYCLE / 2
TASK_SLICE = slice(-(3 + len(M.TASKS)), -3)      # sense(): ..., task one-hot, command speed, sin/cos yaw
STAND_TASK = [1.0 if name == "stand" else 0.0 for name in M.TASKS]
state = {"balanced": False}
plain_sense = agent.sense


def switched_sense(contacts, reference):
    o = plain_sense(contacts, reference)
    if state["balanced"] and not PLAIN and M.one_leg_still(agent.episode_time):
        o[TASK_SLICE] = STAND_TASK
    return o


agent.sense = switched_sense

ticks = round(W.CONTROL_DT / W.TICK)
previous_angles = np.array([s.angle() for s in agent.servos])
pause = None                 # the pause being measured
held = 0.0
rows = []
print(f"mode: {'plain (no switch)' if PLAIN else 'switch to stand once balanced'}, "
      f"{'exploration noise' if NOISE else 'best action'}")
print(" pause starts at  balance found after  then: still  motors deg/s  pelvis cm/s  tilt deg/s")
for step in range(int(seconds / W.TICK)):
    world.update(step % ticks == 0)
    if step % ticks:
        continue
    t = agent.episode_time
    in_pause = M.one_leg_still(t)
    corners = agent._foot_corners()
    down = [c[1] < 0.003 for c in corners]
    touching = [(c[0], c[2]) for c, d in zip(corners, down) if d]
    com, com_v = agent._center_of_mass(), agent._com_velocity()
    margin = support_margin(capture_point((com[0], com[2]), (com_v[0], com_v[2]), com[1]), touching)
    v, av = agent.pelvis.Rigidbody.velocity, agent.pelvis.Rigidbody.angular_velocity
    speed, tilt = math.hypot(v.x, v.z), math.degrees(math.hypot(av.x, av.z))
    angles = np.array([s.angle() for s in agent.servos])
    motors = float(np.mean(np.abs(angles - previous_angles))) / W.CONTROL_DT
    previous_angles = angles

    if in_pause and pause is None:
        pause = {"start": step * W.TICK, "found": None, "waited": 0.0, "n": 0, "ok": 0, "motors": 0.0,
                 "speed": 0.0, "tilt": 0.0}
        state["balanced"], held = False, 0.0
    if pause is None:
        continue
    if not state["balanced"]:
        near = math.hypot(com[0] - agent.com_start_x, com[2] - agent.com_start_z) < BALANCE_COM
        calm = all(down) and speed < BALANCE_SPEED and tilt < BALANCE_TILT and near
        held = held + W.CONTROL_DT if calm else 0.0
        if held >= BALANCE_HOLD:
            state["balanced"] = True
            pause["found"] = step * W.TICK - pause["start"]
        elif t % HALF > M.ONE_LEG_PAUSE - 2 * W.CONTROL_DT and pause["waited"] < WAIT_MAX:
            agent.episode_time -= W.CONTROL_DT       # the clock waits at the end of the pause
            pause["waited"] += W.CONTROL_DT
            continue
    else:
        pause["n"] += 1
        pause["ok"] += int(any(down[:4]) and any(down[4:]) and margin > 0.0 and speed < 0.05)
        pause["motors"] += motors
        pause["speed"] += speed * 100
        pause["tilt"] += tilt
    if not M.one_leg_still(agent.episode_time):      # the pause is over
        n = max(1, pause["n"])
        found = "never" if pause["found"] is None else f"{pause['found']:.2f} s"
        print(f"   {pause['start']:6.1f} s        {found:>8}         "
              f"{pause['ok'] / n:4.0%}     {pause['motors'] / n:5.1f}        {pause['speed'] / n:4.1f}        "
              f"{pause['tilt'] / n:5.1f}")
        rows.append(pause)
        pause = None
        state["balanced"] = False

found = [p for p in rows if p["found"] is not None]
n = sum(p["n"] for p in found)
print(f"pauses: {len(rows)}, balance found in {len(found)}"
      + (f" (after {np.mean([p['found'] for p in found]):.2f} s on average, longest {max(p['found'] for p in found):.2f} s)"
         if found else ""))
if n:
    print(f"after the balance was found: still {sum(p['ok'] for p in found) / n:.0%}, motors "
          f"{sum(p['motors'] for p in found) / n:.1f} deg/s, pelvis {sum(p['speed'] for p in found) / n:.1f} cm/s, "
          f"tilt {sum(p['tilt'] for p in found) / n:.1f} deg/s")
print(f"falls: {agent.falls}")
