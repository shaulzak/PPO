"""
Watch a trained model (no training, best action every step).

    python Run.py                          # model.pt, walking
    python Run.py --stage one_leg          # any stage: stand, shift, one_leg, march, walk
    python Run.py modellatest.pt --pushes  # with random pushes, like in training
    python Run.py --live                   # follow the running training: its newest model and current stage,
                                           # reloaded at the start of every episode
    python Run.py --live --noise --pushes  # what a training episode looks like: the exploration noise too
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bereshit import GameObject, Vector3, Core, Camera
from bereshit.addons.PPO import Academy
from bereshit.addons.essentials import FPS_cam, CamController

import robot as R
from motions import TASKS
from curriculum import STAGES
from WalkAgent import WalkAgent, make_config, check_model_servo, TICK, CONTROL_DT, PHYSICS_EPOCHS

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default=None, help="default model.pt (modellatest.pt with --live)")
parser.add_argument("--stage", choices=TASKS, default=None, help="default walk (the training's stage with --live)")
parser.add_argument("--live", action="store_true", help="follow the running training (see above)")
parser.add_argument("--pushes", action="store_true", help="push the robot now and then, like in training")
parser.add_argument("--noise", action="store_true",
                    help="sample actions with the model's exploration noise, like in training (still no learning)")
args = parser.parse_args()
model_path = os.path.join(HERE, args.model or ("modellatest.pt" if args.live else "model.pt"))
CURRICULUM_STATE = os.path.join(HERE, "curriculum_state.json")


def training_stage():
    try:
        with open(CURRICULUM_STATE, encoding="utf-8") as f:
            return json.load(f).get("stage", "walk")
    except (OSError, ValueError):
        return "walk"


stage = args.stage or (training_stage() if args.live else "walk")

check_model_servo(model_path)
Academy.setup_trainer(make_config(best_model_path=None))
Academy.load_trained_model(model_path)
if args.noise:
    # Still inference mode (nothing stored, no learning), only the action is sampled instead of the best one.
    trainer = Academy.get_trainer()
    best_act = trainer.act
    trainer.act = lambda observation, deterministic=False, **kwargs: best_act(observation, False, **kwargs)
    print(f"exploration noise std {trainer.noise_std():.2f}", flush=True)

root, robot = R.build_robot()
floor = R.build_floor()
agent = WalkAgent(robot, CONTROL_DT, fixed_stage=stage, pushes=args.pushes)
root.add_component(agent)

if args.live:
    loaded = {"mtime": os.path.getmtime(model_path)}
    begin = agent.OnEpisodeBegin

    def follow_training():
        """Before each episode: the training's newest model, and its current stage (unless --stage)."""
        mtime = os.path.getmtime(model_path)
        if mtime != loaded["mtime"]:
            try:
                Academy.load_trained_model(model_path)
                loaded["mtime"] = mtime
                print(f"loaded the model saved at {time.strftime('%H:%M:%S', time.localtime(mtime))}", flush=True)
            except Exception as error:           # the training may be writing the file right now
                print(f"(model not reloaded this time: {error})", flush=True)
        if not args.stage:
            name = training_stage()
            if name != agent.stage.name:
                agent.fixed_stage = next(s for s in STAGES if s.name == name)
                print(f"the training is now on stage '{name}'", flush=True)
        begin()

    agent.OnEpisodeBegin = follow_training
print(f"Watching {os.path.basename(model_path)}, stage '{stage}'" + (" (live)" if args.live else ""), flush=True)
cam = GameObject(position=Vector3(1.2, 0.7, -1.4), rotation=Vector3(15, -40, 0)).add_component(
    Camera(shading="material preview"), CamController(), FPS_cam())

scene = [floor, root, cam]
Core.run(scene, tick=TICK, Render=True, scriptRefreshRate=CONTROL_DT, physics_epochs=PHYSICS_EPOCHS)
