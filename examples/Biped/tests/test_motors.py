"""
Every motor of both legs works during the one-leg stage: follows its command, and the joints that lift the
foot move on each leg in turn (the reference motion alone, no policy). The robot hangs from its pelvis, like
on a test stand: the reference alone doesn't keep the balance on one foot, and every fall would restart the
episode with the same leg first.
    python tests/test_motors.py
"""
import numpy as np
import torch

from _common import check, finish, make_world

from bereshit.addons.PPO import Academy, Trainer
import robot as R
import motions as M
import WalkAgent as W

print("test_motors")


def zero_act(self, observation, deterministic=False, **kwargs):
    return np.zeros(W.ACTION_DIM, dtype=np.float32), None, {
        "value": torch.tensor(0.0), "log_prob": torch.tensor(0.0), "policy_version": 0,
        "observation": torch.zeros(W.OBS_DIM), "continuous_action": torch.zeros(W.ACTION_DIM)}


Trainer.act = zero_act
Academy.setup_trainer(W.make_config(None))
Academy.get_trainer().inference_only = True

root, robot = R.build_robot(lift=0.1)
robot["pelvis"].Rigidbody.isKinematic = True
floor = R.build_floor()
agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage="one_leg", pushes=False, steady_pauses=False)
root.add_component(agent)
world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
ticks = round(W.CONTROL_DT / W.TICK)
for step in range(int(M.ONE_LEG_CYCLE / W.TICK)):
    world.update(step % ticks == 0)

print(f"  episodes {agent.episodes}, falls {agent.falls}, first leg {agent.first_leg}")
m = {k: v / agent.stats.steps for k, v in agent.stats.metrics.items()}
print("\n".join(agent.stats._motor_lines(m, R.JOINT_ORDER)))
names = [(side, j) for side in "LR" for j in R.JOINT_ORDER]
check("one episode, no fall (hanging)", agent.episodes == 1 and agent.falls == 0)
check("all motors of both legs are measured", all(f"motor_error_{s}_{j}" in m for s, j in names))
worst = max(names, key=lambda n: m[f"motor_error_{n[0]}_{n[1]}"])
check("every motor follows its command (average error < 5 deg)", m[f"motor_error_{worst[0]}_{worst[1]}"] < 5.0,
      f"worst {worst[0]} {worst[1]}: {m[f'motor_error_{worst[0]}_{worst[1]}']:.1f} deg")
for j in ("knee_flex", "hip_flex", "hip_roll", "ankle_roll"):
    left, right = m[f"motor_moves_L_{j}"], m[f"motor_moves_R_{j}"]
    check(f"{j} moves on both legs, about equally (one cycle lifts each foot once)",
          min(left, right) > 2.0 and min(left, right) > 0.6 * max(left, right), f"L {left:.1f} R {right:.1f} deg/s")
check("no motor warning on the reference motion", "MOTOR WARNING" not in
      "\n".join(agent.stats._motor_lines(m, R.JOINT_ORDER)))

finish()
