"""
Probe (not part of run_all): the weight-shift reference played open loop, with training-like pushes, for
different shift sizes. Tells whether a stage's demand is doable before blaming the policy.
    python tests/probe_shift_pushes.py
"""
import math
import random
import sys

from _common import make_world

from bereshit import Vector3
import robot as R
import motions as M
import WalkAgent as W


def trial(shift_deg, push, seconds=20.0, seed=0):
    random.seed(seed)
    M.SHIFT_DEG = shift_deg
    root, rb = R.build_robot()
    floor = R.build_floor()
    world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
    pelvis = rb["pelvis"]
    next_push = random.uniform(*W.PUSH_EVERY) + 1.0
    for step in range(int(seconds / W.TICK)):
        t = step * W.TICK
        if step % 4 == 0:
            for s, a in zip(rb["servos"], M.reference_angles("shift", t, 0.0, 0.0)):
                s.command(a)
            if push > 0 and t >= next_push:
                ang = random.uniform(0, 2 * math.pi)
                dv = random.uniform(0.3, 1.0) * push
                v = pelvis.Rigidbody.velocity
                pelvis.Rigidbody.velocity = Vector3(v.x + dv * math.cos(ang), v.y, v.z + dv * math.sin(ang))
                next_push = t + random.uniform(*W.PUSH_EVERY)
        world.update(False)
        up = R.local_to_world(pelvis.transform.quaternion, Vector3(0, 1, 0))
        if pelvis.transform.position.y < W.FALL_HEIGHT or up.y < math.cos(math.radians(W.FALL_TILT_DEG)):
            return t
    return None


for shift in (12.0, 10.0, 8.0):
    for push in (0.0, 0.15):
        results = [trial(shift, push, seed=s) for s in range(3)]
        text = ", ".join("held" if r is None else f"fell {r:.1f}s" for r in results)
        print(f"shift {shift:.0f} deg, pushes {push:.2f} m/s: {text}", flush=True)
