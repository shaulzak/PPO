"""
Train the biped with many simulations at once, one per CPU core.

    python ParallelTrain.py                    # new training, one worker per spare core
    python ParallelTrain.py --resume           # continue from model.pt
    python ParallelTrain.py --workers 6 --seconds 3600

Each worker process runs its own robot with a copy of the policy and collects a slice of the
rollout. The learner (this process) gathers all slices, runs one PPO update on the whole batch,
and sends the new weights back. Same algorithm as Train.py, with N times the experience per minute.
Files are the same as Train.py: model.pt (best), modellatest.pt, eval_log.txt.
"""
import os

# One math thread per process: otherwise every worker's numpy/torch starts a thread per core and
# the workers slow each other down. Must be set before numpy/torch are imported (workers inherit it).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import random
import shutil
import subprocess
import sys
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_PATH = os.path.join(HERE, "model.pt")
EVAL_SNAPSHOT = os.path.join(HERE, "eval_snapshot.pt")
EVAL_EVERY = 5 * 60
REPORT_EVERY = 30
BATCH_STEPS = 4096


def worker(index, steps_per_rollout, conn):
    """Simulates one robot; sends a rollout slice each time the learner sends weights."""
    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    from bereshit import GameObject, Vector3, World
    from bereshit.addons.PPO import Academy
    import robot as R
    from WalkAgent import WalkAgent, make_config, TICK, CONTROL_DT, PHYSICS_EPOCHS

    torch.set_num_threads(1)
    random.seed(1000 + index)
    np.random.seed(1000 + index)
    torch.manual_seed(1000 + index)

    Academy.setup_trainer(make_config(best_model_path=None))
    trainer = Academy.get_trainer()
    trainer.config.rollout_steps = 10 ** 12   # never learn here; the learner does
    trainer.update_obs_stats = False          # the learner owns the normalization statistics

    raw_observations = []
    finished_episodes = []
    act = trainer.act
    record = trainer.record_episode_result

    def act_and_keep(observation, deterministic=False, noise_scale=1.0):
        if not trainer.inference_only:            # like Trainer.act: test episodes don't move the statistics
            raw_observations.append(np.asarray(observation, dtype=np.float32))
        return act(observation, deterministic, noise_scale)

    def record_and_keep(reward, length):
        finished_episodes.append((reward, length))
        record(reward, length)

    trainer.act = act_and_keep
    trainer.record_episode_result = record_and_keep

    root, robot = R.build_robot()
    floor = R.build_floor()
    agent = WalkAgent(robot, CONTROL_DT)
    root.add_component(agent)
    gizmos = GameObject()
    world = World(False, [root, floor], gizmos, Vector3(0, -9.8, 0), TICK, 1, PHYSICS_EPOCHS)
    ticks_per_decision = round(CONTROL_DT / TICK)
    world.Start()
    step = 0

    while True:
        message = conn.recv()
        if message is None:
            return
        weights, rms_state = message
        trainer.model.load_state_dict(weights)
        trainer.obs_rms.load_state_dict(rms_state)
        trainer.policy_version += 1   # a pending action from the old weights gets re-evaluated

        ticks = 0
        while len(trainer.buffer) < steps_per_rollout:
            world.update(step % ticks_per_decision == 0)
            step += 1
            ticks += 1

        conn.send({
            "trajectories": dict(trainer.buffer.agent_trajectories),
            "raw_observations": np.stack(raw_observations) if raw_observations else None,
            "episodes": list(finished_episodes),
            "stats": agent.stats,
            "curriculum": agent.curriculum.report_line(),
            "falls": agent.falls,
            "ticks": ticks,
        })
        trainer.buffer.clear()
        raw_observations.clear()
        finished_episodes.clear()
        agent.stats.reset()


def main():
    import torch
    from bereshit.addons.PPO import Academy
    from WalkAgent import make_config, TICK, CONTROL_DT
    from walk_metrics import WalkStats
    from servos import SERVO
    import robot as R
    curriculum_line = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="continue training from model.pt")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--seconds", type=float, default=None, help="stop after this many wall-clock seconds")
    parser.add_argument("--model", default=MODEL_PATH, help="where to keep the best model (default model.pt)")
    args = parser.parse_args()
    model_path = os.path.abspath(args.model)

    torch.set_num_threads(2)
    Academy.setup_trainer(make_config(model_path))
    trainer = Academy.get_trainer()
    from WalkAgent import mirror_observation, mirror_action
    trainer.set_mirror(mirror_observation, mirror_action)   # symmetry loss between the legs (learner only)
    if args.resume:
        from WalkAgent import check_model_servo
        check_model_servo(model_path)
        Academy.load_model(model_path)
        shutil.copyfile(model_path, os.path.join(HERE, "model_before_resume.pt"))
        trainer.best_average_reward = float("-inf")

    steps_per_rollout = max(64, BATCH_STEPS // args.workers)
    print(f"{args.workers} workers x {steps_per_rollout} decisions per update", flush=True)
    pipes, processes = [], []
    for i in range(args.workers):
        parent, child = mp.Pipe()
        p = mp.Process(target=worker, args=(i, steps_per_rollout, child), daemon=True)
        p.start()
        pipes.append(parent)
        processes.append(p)

    eval_log = open(os.path.join(HERE, "eval_log.txt"), "a", encoding="utf-8")
    stats = WalkStats()
    start = last_report = last_eval = time.perf_counter()
    sim_ticks = 0
    falls = [0] * args.workers
    time_waiting = time_merging = time_learning = 0.0
    try:
        while args.seconds is None or time.perf_counter() - start < args.seconds:
            t0 = time.perf_counter()
            weights = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
            rms_state = trainer.obs_rms.state_dict()
            for conn in pipes:
                conn.send((weights, rms_state))

            for w, conn in enumerate(pipes):
                t_wait = time.perf_counter()
                result = conn.recv()
                t_merge = time.perf_counter()
                time_waiting += t_merge - t_wait
                for agent_id, trajectory in result["trajectories"].items():
                    for transition in trajectory:
                        trainer.buffer.add((w, agent_id), transition)
                trainer.total_environment_steps += sum(len(t) for t in result["trajectories"].values())
                if result["raw_observations"] is not None:
                    trainer.obs_rms.update(result["raw_observations"])
                for reward, length in result["episodes"]:
                    trainer.record_episode_result(reward, length)
                stats.merge(result["stats"])
                curriculum_line = result["curriculum"]   # each worker keeps its own stage progress
                falls[w] = result["falls"]
                sim_ticks += result["ticks"]
                time_merging += time.perf_counter() - t_merge

            t_learn = time.perf_counter()
            trainer.learn_if_ready(force=True)
            time_learning += time.perf_counter() - t_learn

            now = time.perf_counter()
            if now - last_report > REPORT_EVERY:
                sim_time = sim_ticks * TICK
                avg_len = trainer.get_average_episode_length()
                print(f"[{(now - start) / 60:5.1f} min] sim {sim_time / 60:7.1f} min ({sim_time / (now - start):5.1f}x real time, "
                      f"{args.workers} robots) | decisions {trainer.total_environment_steps} | episodes "
                      f"{trainer.completed_episodes} | falls {sum(falls)} | avg episode "
                      f"{'N/A' if avg_len is None else f'{avg_len * CONTROL_DT:.1f} s'}", flush=True)
                print(stats.summary(CONTROL_DT, R.JOINT_ORDER, SERVO, f"worker 0: {curriculum_line}"), flush=True)
                print(f"    time: waiting for robots {time_waiting:.1f} s, merging {time_merging:.1f} s, "
                      f"learning {time_learning:.1f} s", flush=True)
                stats.reset()
                time_waiting = time_merging = time_learning = 0.0
                last_report = now
            if now - last_eval > EVAL_EVERY:
                trainer.save(EVAL_SNAPSHOT)
                subprocess.Popen([sys.executable, os.path.join(HERE, "Evaluate.py"), EVAL_SNAPSHOT, "--seconds", "30",
                                  "--label", f"after {(now - start) / 60:.0f} min"],
                                 stdout=eval_log, stderr=subprocess.DEVNULL)
                last_eval = now
    except KeyboardInterrupt:
        pass
    finally:
        for conn in pipes:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in processes:
            p.join(timeout=5)
    print(f"Stopped. Best model: {model_path}", flush=True)


if __name__ == "__main__":
    main()
