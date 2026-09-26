"""
Is standing on the left foot the mirror image of standing on the right foot, with a trained model? The
one-leg stage's policy sees the mirror image on the left foot, so with a mirror-symmetric robot and physics
the two runs should be mirror images. Prints the pelvis's sideways position over the standing foot and its
roll, side by side (the left-foot run mirrored), and where they part.
    python tests/probe_leg_mirror.py [model.pt] [seconds] [--zero] [--swap]
    --zero: the reference motion only, no policy (does the engine itself treat the two legs the same?)
    --swap: build the right leg before the left (the engine solves bodies/joints/contacts in build order)
"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from bereshit import GameObject, Vector3, World
from bereshit.addons.PPO import Academy
from bereshit.addons.PPO.PPO import Trainer

import robot as R
import motions as M
from WalkAgent import WalkAgent, make_config, TICK, CONTROL_DT, PHYSICS_EPOCHS, ACTION_DIM, OBS_DIM

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model = args[0] if args else os.path.join(HERE, "modellatest.pt")
seconds = float(args[1]) if len(args) > 1 else M.ONE_LEG_CYCLE / 2
Academy.setup_trainer(make_config(best_model_path=None))
if "--zero" in sys.argv:
    def zero_act(self, observation, deterministic=False, **kwargs):
        return np.zeros(ACTION_DIM, dtype=np.float32), None, {
            "value": torch.tensor(0.0), "log_prob": torch.tensor(0.0), "policy_version": 0,
            "observation": torch.zeros(OBS_DIM), "continuous_action": torch.zeros(ACTION_DIM)}
    Trainer.act = zero_act
    Academy.get_trainer().inference_only = True
    model = "zero actions"
else:
    Academy.load_trained_model(model)

KEEP = []   # the engine frees components whose Python objects die


def run(stance):
    """stance 0: stand on the left foot (the right lifts first), 1: on the right. Samples every 0.1 s."""
    root, robot = R.build_robot()
    if "--swap" in sys.argv:
        pelvis, leg_l, leg_r = root.children
        KEEP.append(root)
        root = GameObject(size=Vector3(), children=[pelvis, leg_r, leg_l], name="biped")
    floor = R.build_floor()
    agent = WalkAgent(robot, CONTROL_DT, fixed_stage="one_leg", pushes=False)
    root.add_component(agent)
    world = World(False, [root, floor], GameObject(), Vector3(0, -9.8, 0), TICK, 1, PHYSICS_EPOCHS)
    world.Start()
    agent.begin_episode()
    agent.first_leg, agent.only_stance = 1 - stance, None
    mirror = 1.0 if stance == 1 else -1.0      # show both as "standing on the right foot"
    side = "left" if stance == 0 else "right"
    samples, falls = [], agent.falls
    ticks = round(CONTROL_DT / TICK)
    for step in range(int(seconds / TICK)):
        world.update(step % ticks == 0)
        if agent.falls > falls:
            samples.append(None)
            break
        if step % round(0.1 / TICK) == 0:
            pelvis = robot["pelvis"].transform
            foot = robot[side]["foot"].transform.position
            up = R.local_to_world(pelvis.quaternion, Vector3(0, 1, 0))
            samples.append((mirror * (pelvis.position.z - foot.z) * 100, math.degrees(math.asin(max(-1, min(1, mirror * up.z)))),
                            pelvis.position.x * 100))
    return samples


left, right = run(0), run(1)
print(f"{model}: pelvis over the standing foot, cm (sideways) / pelvis roll, deg / forward, cm")
print("   t  | on the left (mirrored)  | on the right          | difference")
for i in range(max(len(left), len(right))):
    a = left[i] if i < len(left) else None
    b = right[i] if i < len(right) else None
    fmt = lambda s: "fell" if s is None else f"{s[0]:+6.2f} {s[1]:+6.2f} {s[2]:+6.2f}"
    diff = "" if a is None or b is None else f"{a[0] - b[0]:+6.2f} {a[1] - b[1]:+6.2f} {a[2] - b[2]:+6.2f}"
    print(f"{i * 0.1:5.1f} | {fmt(a):22} | {fmt(b):22} | {diff}")
