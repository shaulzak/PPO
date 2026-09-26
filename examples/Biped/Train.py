"""
Train the biped through the stages in curriculum.py (stand -> shift -> one leg -> march -> walk),
headless and as fast as the CPU allows.

    python Train.py                 # new training from the first stage
    python Train.py --resume        # continue from model.pt at the stage reached (curriculum_state.json)
    python Train.py --seconds 600   # stop after 10 minutes

Stop any time with Ctrl+C: model.pt always holds the best model so far (modellatest.pt the newest).
Every 5 minutes the deterministic policy is tested on the current stage (eval_log.txt). Watch with Run.py.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bereshit import GameObject, Vector3, World
from bereshit.addons.PPO import Academy

import robot as R
from curriculum import Curriculum
from WalkAgent import (WalkAgent, make_config, check_model_servo, mirror_observation, mirror_action,
                       TICK, CONTROL_DT, PHYSICS_EPOCHS)
from servos import SERVO

MODEL_PATH = os.path.join(HERE, "model.pt")
CURRICULUM_STATE = os.path.join(HERE, "curriculum_state.json")

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="continue training from model.pt")
parser.add_argument("--seconds", type=float, default=None, help="stop after this many wall-clock seconds")
args = parser.parse_args()

print(f"Servo: {SERVO.name} ({SERVO.stall_torque:.2f} N*m, {SERVO.max_speed:.0f} deg/s, "
      f"position feedback: {'yes' if SERVO.position_feedback else 'no'}) | {R.NUM_SERVOS} servos, "
      f"hip yaw: {'yes' if R.HIP_YAW else 'no'}, leg channel {R.LEG_CHANNEL}, battery: {'yes' if R.BATTERY else 'no'}",
      flush=True)
if not args.resume:
    # A new training starts from the first stage, with no time or test history.
    for old in (CURRICULUM_STATE, os.path.join(HERE, "stage_progress.csv")):
        if os.path.exists(old):
            os.remove(old)
curriculum = Curriculum(CURRICULUM_STATE)
print(f"Starting at stage {curriculum.level + 1}: {curriculum.stage.name}", flush=True)
Academy.setup_trainer(make_config(MODEL_PATH))
trainer = Academy.get_trainer()
trainer.set_mirror(mirror_observation, mirror_action)   # symmetry loss between the legs
if args.resume:
    check_model_servo(MODEL_PATH)
    # Continue from where training stopped: modellatest.pt when it is further on than model.pt (the best
    # average reward, which after a reward change can be an older point of the run).
    LATEST_PATH = MODEL_PATH[:-3] + "latest.pt"
    resume_from = MODEL_PATH
    if os.path.exists(LATEST_PATH):
        import torch
        updates = lambda p: torch.load(p, map_location="cpu", weights_only=False).get("training_updates", 0)
        if updates(LATEST_PATH) > updates(MODEL_PATH):
            resume_from = LATEST_PATH
    print(f"Resuming from {os.path.basename(resume_from)}", flush=True)
    Academy.load_model(resume_from)
    if resume_from != MODEL_PATH:
        shutil.copyfile(resume_from, MODEL_PATH)
    # Keep the model we resumed from, and let model.pt follow the new run: its best score was
    # measured with whatever reward was used then, which may not be comparable.
    shutil.copyfile(MODEL_PATH, os.path.join(HERE, "model_before_resume.pt"))
    trainer.best_average_reward = float("-inf")

# Exploration noise: brought down step by step to NOISE_TARGET over NOISE_DECAY_UPDATES updates from where
# it is now. At ~0.31 (~6 deg per servo) the one-leg stage fell during training and never learned to stand
# still (the robot trembled at ~5 Hz); the pass tests run without noise anyway. Only when resuming: a new
# training from the first stage needs its full exploration.
# While it should stand still the noise is much lower (WalkAgent.STILL_NOISE_STD), everywhere else this.
NOISE_TARGET = 0.15
NOISE_DECAY_UPDATES = 40
noise_start, noise_from_update = max(NOISE_TARGET, trainer.noise_std()), trainer.training_updates


def noise_cap(updates):
    done = min(1.0, max(0, updates - noise_from_update) / NOISE_DECAY_UPDATES)
    return noise_start + (NOISE_TARGET - noise_start) * done if args.resume else None


trainer.set_max_noise_std(noise_cap(trainer.training_updates))
noise_updates = trainer.training_updates

root, robot = R.build_robot()
floor = R.build_floor()
agent = WalkAgent(robot, CONTROL_DT, curriculum=curriculum)
root.add_component(agent)
# Every scene object must stay referenced from Python, or the engine ends up with freed components.
gizmos = GameObject()
scene = [root, floor]
world = World(False, scene, gizmos, Vector3(0, -9.8, 0), TICK, 1, PHYSICS_EPOCHS)

EVAL_EVERY = 5 * 60   # s: test the deterministic policy (what Run.py shows) in a separate process
EVAL_SNAPSHOT = os.path.join(HERE, "eval_snapshot.pt")
eval_log = open(os.path.join(HERE, "eval_log.txt"), "a", encoding="utf-8")


def launch_evaluation(minutes):
    trainer.save(EVAL_SNAPSHOT)
    subprocess.Popen([sys.executable, os.path.join(HERE, "Evaluate.py"), EVAL_SNAPSHOT, "--seconds", "30",
                      "--stage", curriculum.stage.name, "--label", f"after {minutes:.0f} min"],
                     stdout=eval_log, stderr=subprocess.DEVNULL)


ticks_per_decision = round(CONTROL_DT / TICK)
world.Start()
start = time.perf_counter()
last_report = start
last_eval = start
level = curriculum.level
step = 0
try:
    while args.seconds is None or time.perf_counter() - start < args.seconds:
        world.update(step % ticks_per_decision == 0)
        step += 1
        if trainer.training_updates != noise_updates:
            noise_updates = trainer.training_updates
            trainer.set_max_noise_std(noise_cap(noise_updates))
        if curriculum.level != level:
            # Passed a stage: keep that model, and restart the "best" comparison (rewards differ per stage).
            trainer.save(os.path.join(HERE, f"model_passed_{curriculum.STAGES_NAMES[level]}.pt"))
            trainer.best_average_reward = float("-inf")
            trainer.episode_rewards.clear()
            level = curriculum.level
        now = time.perf_counter()
        if now - last_report > 30:
            sim_time = step * TICK
            avg_len = trainer.get_average_episode_length()
            print(f"[{(now - start) / 60:5.1f} min] sim {sim_time / 60:6.1f} min ({sim_time / (now - start):4.1f}x real time) | "
                  f"decisions {trainer.total_environment_steps} | episodes {trainer.completed_episodes} | "
                  f"falls {agent.falls} | avg episode "
                  f"{'N/A' if avg_len is None else f'{avg_len * CONTROL_DT:.1f} s'} | exploration noise std "
                  f"{trainer.noise_std():.2f}" + ("" if trainer.config.max_noise_std is None else
                                                f" (at most {trainer.config.max_noise_std:.2f})"), flush=True)
            print(agent.stats.summary(CONTROL_DT, R.JOINT_ORDER, SERVO, curriculum.report_line()), flush=True)
            agent.stats.reset()
            last_report = now
        if now - last_eval > EVAL_EVERY:
            launch_evaluation((now - start) / 60)
            last_eval = now
except KeyboardInterrupt:
    pass
print(f"Stopped. Best model: {MODEL_PATH}")
