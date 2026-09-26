"""
Probe: how each stage's reward scores its own reference motion, played with zero corrections.
A stage whose reference scores low or negative is teaching the policy to move away from it.
    python tests/probe_reference_reward.py [stage ...]
"""
import sys

from _common import make_world

import numpy as np
import torch
from bereshit.addons.PPO import Academy, Trainer

import robot as R
import motions as M
from servos import SERVO
import WalkAgent as W


def zero_act(self, observation, deterministic=False, **kwargs):
    zeros = np.zeros(W.ACTION_DIM, dtype=np.float32)
    return zeros, None, {"value": torch.tensor(0.0), "log_prob": torch.tensor(0.0), "policy_version": 0,
                         "observation": torch.zeros(W.OBS_DIM), "continuous_action": torch.zeros(W.ACTION_DIM)}


Trainer.act = zero_act
Academy.setup_trainer(W.make_config(None))
Academy.get_trainer().inference_only = True

stages = sys.argv[1:] or M.TASKS
for name in stages:
    root, robot = R.build_robot()
    floor = R.build_floor()
    agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage=name, pushes=True)
    root.add_component(agent)
    world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
    # fixed_stage runs record no stats while is_test is False, so collect them explicitly
    for step in range(int(20 / W.TICK)):
        world.update(step % 4 == 0)
    print(f"=== reference '{name}' with zero corrections: {agent.falls} falls in 20 s")
    print(agent.stats.summary(W.CONTROL_DT, R.JOINT_ORDER, SERVO), flush=True)
