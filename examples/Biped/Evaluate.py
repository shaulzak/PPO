"""
Measure what a model really does, the way Run.py runs it (best action, no training), without a window.

    python Evaluate.py model.pt                          # 30 s of walking
    python Evaluate.py model.pt --stage one_leg --seconds 60
    python Evaluate.py model.pt --push-test --stage stand # strongest push it survives, per direction

Train.py runs this every few minutes on a snapshot and appends the result to eval_log.txt.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bereshit import GameObject, Vector3, World
from bereshit.addons.PPO import Academy

import robot as R
from motions import TASKS
from servos import SERVO
from WalkAgent import WalkAgent, make_config, check_model_servo, TICK, CONTROL_DT, PHYSICS_EPOCHS

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default=os.path.join(HERE, "model.pt"))
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--stage", choices=TASKS, default="walk")
parser.add_argument("--label", default=None)
parser.add_argument("--push-test", action="store_true",
                    help="push the pelvis harder and harder in 4 directions; report the strongest survived")
args = parser.parse_args()
label = args.label or os.path.basename(args.model)

check_model_servo(args.model)
Academy.setup_trainer(make_config(best_model_path=None))
Academy.load_trained_model(args.model)
root, robot = R.build_robot()
floor = R.build_floor()
agent = WalkAgent(robot, CONTROL_DT, fixed_stage=args.stage, pushes=False)
root.add_component(agent)
gizmos = GameObject()
world = World(False, [root, floor], gizmos, Vector3(0, -9.8, 0), TICK, 1, PHYSICS_EPOCHS)
ticks_per_decision = round(CONTROL_DT / TICK)
world.Start()
step = 0


def run(seconds):
    global step
    for _ in range(int(seconds / TICK)):
        world.update(step % ticks_per_decision == 0)
        step += 1


if not args.push_test:
    run(args.seconds)
    print(f"EVAL {label}, stage '{args.stage}': {agent.falls} falls in {args.seconds:.0f} s (best action, like Run.py)")
    print(agent.stats.summary(CONTROL_DT, R.JOINT_ORDER, SERVO), flush=True)
else:
    # Each trial: a fresh episode, 2 s to settle, one push of dv (m/s) to the pelvis, 3 s to recover.
    directions = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1)}
    results = {}
    for name, (dx, dz) in directions.items():
        survived = 0.0
        for dv in [0.1 * k for k in range(1, 31)]:
            agent.begin_episode()
            falls_before = agent.falls
            run(2.0)
            agent.give_push(dv * dx, dv * dz)
            run(3.0)
            if agent.falls > falls_before:
                break
            survived = dv
        results[name] = survived
    pelvis_mass = robot["pelvis"].Rigidbody.mass
    print(f"PUSH TEST {label}, stage '{args.stage}': strongest push survived (pelvis speed change, m/s; "
          f"x{pelvis_mass:.2f} kg = impulse N*s): " + ", ".join(
              f"{k} {v:.1f} ({v * pelvis_mass:.2f} N*s)" for k, v in results.items()), flush=True)
