"""
Are the reference motions physically doable? Each stage's reference is played open-loop (no policy, no
corrections) on the standing robot. Standing and the slow weight shift should hold on their own; one-leg
and stepping need balance corrections, so for those this only reports how long they last.
    python tests/test_reference_motions.py
"""
import math

from _common import check, finish, make_world

from bereshit import Vector3
import robot as R
import motions as M
import WalkAgent as W

print("test_reference_motions (open loop)")


def play(task, seconds, **overrides):
    saved = {k: getattr(M, k) for k in overrides}
    for k, v in overrides.items():
        setattr(M, k, v)
    root, rb = R.build_robot()
    floor = R.build_floor()
    world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
    bodies = [rb["pelvis"]] + list(rb["left"].values()) + list(rb["right"].values())
    masses = [b.Rigidbody.mass for b in bodies]
    com_z = []
    fell = None
    for step in range(int(seconds / W.TICK)):
        t = step * W.TICK
        if step % 4 == 0:
            for s, a in zip(rb["servos"], M.reference_angles(task, t, (t / M.GAIT_PERIOD) % 1.0, 0.05)):
                s.command(a)
        world.update(False)
        up = R.local_to_world(rb["pelvis"].transform.quaternion, Vector3(0, 1, 0))
        if rb["pelvis"].transform.position.y < W.FALL_HEIGHT or up.y < math.cos(math.radians(W.FALL_TILT_DEG)):
            fell = t
            break
        com_z.append(sum(m * b.transform.position.z for m, b in zip(masses, bodies)) / sum(masses))
    for k, v in saved.items():
        setattr(M, k, v)
    swing = (max(com_z) - min(com_z)) / 2 if com_z else 0.0
    return fell, swing


fell, _ = play("stand", 5)
check("stand: holds 5 s", fell is None, "fell at %.1f s" % fell if fell else "")
fell, swing = play("shift", 2 * M.SHIFT_PERIOD)
check(f"shift ({M.SHIFT_DEG:.0f} deg, {M.SHIFT_PERIOD:.0f} s): holds two cycles", fell is None,
      ("fell at %.1f s" % fell if fell else "ok") + f", center of mass swings +-{swing * 100:.1f} cm")
from curriculum import STAGES
one_leg_stage = next(s for s in STAGES if s.name == "one_leg")
check("one-leg episode holds each foot up twice", one_leg_stage.seconds >= 2 * M.ONE_LEG_CYCLE,
      f"episode {one_leg_stage.seconds} s, cycle {M.ONE_LEG_CYCLE:.1f} s")
check("one-leg lift and lower are slow (>= 1.5 s)", M.ONE_LEG_LIFT >= 1.5 and M.ONE_LEG_LOWER >= 1.5)
half = M.ONE_LEG_CYCLE / 2
check("standing still is measured from the settle time to the end of each pause, in both halves",
      all(not M.one_leg_still(h + W.ONE_LEG_SETTLE - 0.01, W.ONE_LEG_SETTLE)
          and M.one_leg_still(h + W.ONE_LEG_SETTLE + 0.01, W.ONE_LEG_SETTLE)
          and not M.one_leg_still(h + M.ONE_LEG_PAUSE + 0.01, W.ONE_LEG_SETTLE) for h in (0.0, half))
      and W.ONE_LEG_SETTLE <= M.ONE_LEG_PAUSE / 2)
check("each one-leg pause has >= 2 s of measured standing still, after a slow (>= 1.5 s) weight-back",
      M.ONE_LEG_PAUSE - W.ONE_LEG_SETTLE >= 2.0 and M.ONE_LEG_BACK >= 1.5)
# Every reference is the same on both legs: the left leg's targets in one run are the right leg's in the
# mirror-image run (angles are in each leg's own convention; the servo's sign mirrors the roll/yaw joints).
# The one-leg hip hike used to be 8 deg apart between standing on the left and on the right.
n = R.JOINTS_PER_LEG
swap = lambda a: list(a[n:]) + list(a[:n])
worst = {}
for k in range(int(M.ONE_LEG_CYCLE / 0.01)):
    t = k * 0.01
    pairs = {"one_leg": (M.reference_angles("one_leg", t, 0.0, 0.0, first_leg=1),
                         M.reference_angles("one_leg", t, 0.0, 0.0, first_leg=0)),
             "shift": (M.reference_angles("shift", t, 0.0, 0.0),
                       M.reference_angles("shift", t + M.SHIFT_PERIOD / 2, 0.0, 0.0))}
    phase = (t / M.GAIT_PERIOD) % 1.0
    for task in ("march", "walk"):   # t past the start ramp; half a gait cycle later the legs swap
        pairs[task] = (M.reference_angles(task, 10.0, phase, 0.05), M.reference_angles(task, 10.0, (phase + 0.5) % 1.0, 0.05))
    for task, (a, b) in pairs.items():
        worst[task] = max(worst.get(task, 0.0), max(abs(x - y) for x, y in zip(a, swap(b))))
for task, gap in worst.items():
    check(f"{task}: the reference is the mirror image on the other leg", gap < 1e-9, f"largest gap {gap:.4f} deg")

for task, seconds in (("one_leg", M.ONE_LEG_CYCLE), ("march", 3 * M.GAIT_PERIOD)):
    fell, _ = play(task, seconds)
    print(f"  [INFO] {task}: open loop {'fell at %.1f s' % fell if fell else 'held'} (needs balance corrections)")

finish()
