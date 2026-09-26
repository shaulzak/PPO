"""
The simulated robot itself: standing still, following servo targets, hinge alignment, swing-foot clearance.
    python tests/test_robot.py
"""
import math
import time

from _common import check, finish, make_world

from bereshit import Vector3
import robot as R
import motions as M
import WalkAgent as W

print(f"test_robot ({R.SERVO_MODEL}, hip yaw {'on' if R.HIP_YAW else 'off'}, {R.LEG_CHANNEL} legs)")


def build(lift, hang):
    root, rb = R.build_robot(lift=lift)
    if hang:
        rb["pelvis"].Rigidbody.isKinematic = True
    floor = R.build_floor()
    return make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS), rb


def pose(rb, targets=None):
    for i, s in enumerate(rb["servos"]):
        joint = R.JOINT_ORDER[i % R.JOINTS_PER_LEG]
        s.command((targets or {}).get(joint, M.DEFAULT_POSE[joint]))


def tilt_deg(body):
    up = R.local_to_world(body.transform.quaternion, Vector3(0, 1, 0))
    return math.degrees(math.acos(max(-1.0, min(1.0, up.y))))


# 1. standing in the default pose holds still
world, rb = build(0.002, False)
pose(rb)
t0 = time.perf_counter()
for step in range(1000):
    world.update(step % 4 == 0)
speed = 5 / (time.perf_counter() - t0)
pelvis = rb["pelvis"]
check("stands 5 s in the default pose", pelvis.transform.position.y > 0.6 and tilt_deg(pelvis) < 3.0,
      f"pelvis y {pelvis.transform.position.y:.3f} m, tilt {tilt_deg(pelvis):.1f} deg")
feet_speed = max(rb[s]["foot"].Rigidbody.velocity.magnitude() for s in ("left", "right"))
check("feet at rest after 5 s", feet_speed < 0.01, f"{feet_speed * 100:.2f} cm/s")
check("simulation runs faster than 5x real time", speed > 5, f"{speed:.1f}x")

# 2. hanging: joints reach their targets and hinges keep their axes
world, rb = build(0.3, True)
targets = {"knee_flex": 90.0, "hip_flex": 30.0, "ankle_pitch": 20.0}
pose(rb, {j: targets.get(j, 0.0) for j in R.JOINT_ORDER})
worst_leak = 0.0
for step in range(400):
    world.update(step % 4 == 0)
    if step > 100:
        for s in rb["servos"]:
            a = R.local_to_world(s.mount.transform.quaternion, s.axis_local).normalized()
            b = R.local_to_world(s.body.transform.quaternion, s.axis_local).normalized()
            worst_leak = max(worst_leak, math.degrees(math.acos(max(-1.0, min(1.0, a.dot(b))))))
errors = {j: abs(rb["servos"][R.servo_index(j, 0)].angle() - targets.get(j, 0.0)) for j in R.JOINT_ORDER}
check("hanging joints reach their targets within 2 deg", max(errors.values()) < 2.0,
      ", ".join(f"{j} {e:.1f}" for j, e in errors.items()))
check("hinges keep their axis (< 0.5 deg)", worst_leak < 0.5, f"worst {worst_leak:.2f} deg")

# 3. the reference gait lifts the swinging foot and keeps it level
world, rb = build(0.3, True)
foot = rb["left"]["foot"]
hx, hy, hz = R.FOOT_LENGTH / 2, R.FOOT_THICKNESS / 2, R.FOOT_WIDTH / 2


def lowest_corner():
    q, p = foot.transform.quaternion, foot.transform.position
    return min((p + R.local_to_world(q, Vector3(cx, -hy, cz))).y for cx in (hx, -hx) for cz in (hz, -hz))


base, best, pitch_at_best = None, -1.0, 0.0
for step in range(int(3 * M.GAIT_PERIOD / W.TICK)):
    t = step * W.TICK
    phase = (t / M.GAIT_PERIOD) % 1.0
    if step % 4 == 0:
        for s, a in zip(rb["servos"], M.reference_angles("march", 10.0, phase, 0.0)):
            s.command(a)
    world.update(False)
    if t > M.GAIT_PERIOD and phase < 0.01 and base is None:
        base = lowest_corner()
    if base is not None and 0.15 < phase < 0.35:
        clear = lowest_corner() - base
        if clear > best:
            f = R.local_to_world(foot.transform.quaternion, Vector3(1, 0, 0))
            best, pitch_at_best = clear, math.degrees(math.atan2(f.y, math.hypot(f.x, f.z)))
check("swing foot rises >= 5 cm in mid-swing", best >= 0.05, f"{best * 100:.1f} cm")
check("swing foot stays level (< 5 deg)", abs(pitch_at_best) < 5.0, f"{pitch_at_best:+.1f} deg")

# 4. the two legs are mirror images: every part's height, spacing, size, mass, inertia and rotation, and every
#    servo's settings (only the roll/yaw servos turn the opposite way, which is what mirroring needs)
root, rb = R.build_robot()
diffs = []
for name in rb["left"]:
    a, b = rb["left"][name], rb["right"][name]
    pa, pb = a.transform.position, b.transform.position
    sa, sb = a.transform.size, b.transform.size
    qa, qb = a.transform.quaternion, b.transform.quaternion
    if max(abs(pa.x - pb.x), abs(pa.y - pb.y), abs(pa.z + pb.z)) > 1e-9:
        diffs.append(f"{name} position")
    if (sa.x, sa.y, sa.z) != (sb.x, sb.y, sb.z):
        diffs.append(f"{name} size")
    ia, ib = a.Rigidbody.inertia, b.Rigidbody.inertia
    if a.Rigidbody.mass != b.Rigidbody.mass or (ia.x, ia.y, ia.z) != (ib.x, ib.y, ib.z):
        diffs.append(f"{name} mass/inertia")
    # mirror across z: (w, x, y, z) -> (w, -x, -y, z)
    if max(abs(qa.w - qb.w), abs(qa.x + qb.x), abs(qa.y + qb.y), abs(qa.z - qb.z)) > 1e-9:
        diffs.append(f"{name} rotation")
check("both legs have the same parts", sorted(rb["left"]) == sorted(rb["right"]), f"{sorted(rb['left'])}")
check("legs' parts are mirror images", not diffs, ", ".join(diffs) or f"{len(rb['left'])} parts each")
pelvis = rb["pelvis"].transform.position
check("pelvis (with the battery) is centred", abs(pelvis.z) < 1e-9, f"z {pelvis.z * 100:+.3f} cm")
servo_diffs = []
for i in range(R.JOINTS_PER_LEG):
    joint = R.JOINT_ORDER[i]
    a, b = vars(rb["servos"][R.servo_index(joint, 0)]), vars(rb["servos"][R.servo_index(joint, 1)])
    for key, value in a.items():
        if not isinstance(value, (int, float)):
            continue
        want = -value if key == "sign" and ("roll" in joint or "yaw" in joint) else value
        if b.get(key) != want:
            servo_diffs.append(f"{joint}.{key} {value}/{b.get(key)}")
check("servos match on both legs (roll/yaw mirrored)", not servo_diffs, ", ".join(servo_diffs) or
      f"{R.JOINTS_PER_LEG} joints")

finish()
