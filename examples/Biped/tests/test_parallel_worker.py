"""
ParallelTrain's worker, run in this process with a fake pipe: it must collect a rollout slice, and keep only
the observations of training actions for the learner's normalization statistics.
    python tests/test_parallel_worker.py
"""
import os
import threading

import numpy as np

from _common import check, finish

from bereshit.addons.PPO import Academy
import ParallelTrain as P

print("test_parallel_worker")

STEPS = 64
MARK = 7.25   # an observation no robot produces


class FakeConn:
    """The learner's end of the pipe: sends the weights once, then asks the worker to stop."""

    def __init__(self):
        self.sent = []
        self.received = 0

    def recv(self):
        self.received += 1
        if self.received > 1:
            return None
        trainer = Academy.get_trainer()
        # an action taken in a test episode (inference_only): it must not reach the learner's statistics
        trainer.inference_only = True
        trainer.act(np.full(trainer.config.obs_dim, MARK, dtype=np.float32))
        trainer.inference_only = False
        return trainer.model.state_dict(), trainer.obs_rms.state_dict()

    def send(self, message):
        self.sent.append(message)


def give_up():
    # The engine catches an exception in Update and goes on, so a worker whose every action fails never fills
    # its rollout: it spins forever and the learner waits for it forever.
    check("the worker collects a rollout slice without crashing", False, f"no slice after {TIMEOUT} s")
    finish()


TIMEOUT = 120
watchdog = threading.Timer(TIMEOUT, lambda: (give_up(), os._exit(1)))
watchdog.daemon = True
watchdog.start()

conn = FakeConn()
error = None
try:
    P.worker(0, STEPS, conn)
except Exception as e:   # the worker's act wrapper didn't take noise_scale: TypeError on the first action
    error = e
watchdog.cancel()

check("the worker collects a rollout slice without crashing", error is None and len(conn.sent) == 1,
      repr(error) if error else f"{len(conn.sent)} slice(s) sent")
raw = conn.sent[0]["raw_observations"] if conn.sent else None
check("the slice brings the raw observations of its training actions", raw is not None and len(raw) >= STEPS,
      f"{0 if raw is None else len(raw)} observations for {STEPS} steps")
check("an action in a test episode is left out of the statistics",
      raw is not None and not np.any(np.all(raw == MARK, axis=1)),
      "no rollout" if raw is None else f"{int(np.all(raw == MARK, axis=1).sum())} test observations kept")

finish()
