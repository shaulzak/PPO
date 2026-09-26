"""
Where does the turning in place come from? Runs a trained model standing (or shifting) and prints, every
second, the heading of the pelvis and of each foot since the start, and which feet touch the floor: if the
feet keep their heading and only the pelvis turns, it's the hip-yaw servos; if the feet turn too, they slip.
    python tests/probe_turning.py [model.pt | zero] [stage] [seconds]
"zero" plays the reference motion with no corrections (no network): turning then is the physics' own.
"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from bereshit import GameObject, Vector3, World
from bereshit.addons.PPO import Academy, Trainer

import robot as R
from WalkAgent import WalkAgent, make_config, TICK, CONTROL_DT, PHYSICS_EPOCHS, OBS_DIM, ACTION_DIM

model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "model.pt")
stage = sys.argv[2] if len(sys.argv) > 2 else "stand"
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
if model == "zero":
    Trainer.act = lambda self, observation, deterministic=False, **kwargs: (np.zeros(ACTION_DIM, dtype=np.float32), None, {
        "value": torch.tensor(0.0), "log_prob": torch.tensor(0.0), "policy_version": 0,
        "observation": torch.zeros(OBS_DIM), "continuous_action": torch.zeros(ACTION_DIM)})
    Academy.setup_trainer(make_config(None))
    Academy.get_trainer().inference_only = True
else:
    Academy.setup_trainer(make_config(best_model_path=None))
    Academy.load_trained_model(model)


def heading(body):
    forward = R.local_to_world(body.transform.quaternion, Vector3(1, 0, 0))
    return math.degrees(math.atan2(forward.z, forward.x))


root, robot = R.build_robot()
floor = R.build_floor()
agent = WalkAgent(robot, CONTROL_DT, fixed_stage=stage, pushes=False)
root.add_component(agent)
world = World(False, [root, floor], GameObject(), Vector3(0, -9.8, 0), TICK, 1, PHYSICS_EPOCHS)
world.Start()
agent.begin_episode()
parts = [robot["pelvis"], robot["left"]["foot"], robot["right"]["foot"]]
start = [heading(p) for p in parts]
ticks = round(CONTROL_DT / TICK)
print(f"{model}, stage '{stage}': heading since the start, deg (+ = to the left)")
print("   t  | pelvis | left foot | right foot | feet down (L R)")
for step in range(int(seconds / TICK) + 1):
    if step % round(1.0 / TICK) == 0:
        now = [(heading(p) - s + 180) % 360 - 180 for p, s in zip(parts, start)]
        corners = agent._foot_corners()
        down = ["x" if min(c[1] for c in corners[4 * i:4 * i + 4]) < 0.003 else "." for i in (0, 1)]
        print(f"{step * TICK:5.1f} | {now[0]:+6.1f} | {now[1]:+9.1f} | {now[2]:+10.1f} | {down[0]} {down[1]}")
    falls = agent.falls
    world.update(step % ticks == 0)
    if agent.falls > falls:
        print(f"fell at {step * TICK:.1f} s")
        break
