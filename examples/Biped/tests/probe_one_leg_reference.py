"""
Probe (not in run_all): the one-leg reference played open loop, first half cycle (left foot lifts). Where is
the center of mass relative to the standing foot when the lift starts, how high does the lifted foot get,
and what tips the robot over. Answers "is the reference itself doable, or does it ask for a lift before
the weight is over the standing foot?".
    python tests/probe_one_leg_reference.py [one_leg_shift_deg [hike_deg]]
"""
import math
import sys

from _common import make_world

from bereshit import Vector3
import robot as R
import motions as M
import WalkAgent as W

if len(sys.argv) > 1:
    M.ONE_LEG_SHIFT_DEG = float(sys.argv[1])
if len(sys.argv) > 2:
    M.ONE_LEG_HIKE_DEG = float(sys.argv[2])
root, rb = R.build_robot()
floor = R.build_floor()
world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
bodies = [rb["pelvis"]] + list(rb["left"].values()) + list(rb["right"].values())
masses = [b.Rigidbody.mass for b in bodies]
feet = [rb["left"]["foot"], rb["right"]["foot"]]


def corners(foot):
    q, p = foot.transform.quaternion, foot.transform.position
    hx, hy, hz = R.FOOT_LENGTH / 2, R.FOOT_THICKNESS / 2, R.FOOT_WIDTH / 2
    return [p + R.local_to_world(q, Vector3(cx, -hy, cz)) for cx, cz in ((hx, hz), (hx, -hz), (-hx, hz), (-hx, -hz))]


print(f"ONE_LEG_SHIFT_DEG {M.ONE_LEG_SHIFT_DEG}, knee lift {M.ONE_LEG_KNEE_LIFT_DEG}, hip lift {M.ONE_LEG_HIP_LIFT_DEG}, hike {M.ONE_LEG_HIKE_DEG}, pause {M.ONE_LEG_PAUSE} s, lean {M.ONE_LEG_LEAN} s, lift {M.ONE_LEG_LIFT} s")
print("   t  lean  lift | CoM over right foot: z from its center (cm, + = toward its inner edge) | "
      "left foot up (cm) | pelvis roll")
half = M.ONE_LEG_CYCLE / 2
for step in range(int(half / W.TICK)):
    t = step * W.TICK
    if step % 4 == 0:
        for s, a in zip(rb["servos"], M.reference_angles("one_leg", t, 0.0, 0.0, first_leg=0)):
            s.command(a)
    world.update(False)
    if step % int(0.2 / W.TICK):
        continue
    com = sum((b.transform.position * m for m, b in zip(masses, bodies)), Vector3(0, 0, 0)) * (1 / sum(masses))
    right = feet[1].transform.position
    up = R.local_to_world(rb["pelvis"].transform.quaternion, Vector3(0, 1, 0))
    lean, lifted, lift = M.one_leg_schedule(t, 0)
    left_up = min(c.y for c in corners(feet[0])) - R.FLOOR_TOP
    roll = math.degrees(math.atan2(up.z, up.y))
    # z grows toward the left foot here (left foot is at +z): distance of the CoM from the right foot center
    stance = {j: rb["servos"][R.servo_index(j, 1)] for j in ("hip_roll", "ankle_roll", "knee_flex")}
    motors = " ".join(f"{j} {s.angle() - s.command_deg:+5.1f}deg {abs(s.joint.motor_impulse / W.TICK) / s.max_torque:4.0%}"
                      for j, s in stance.items())
    print(f"{t:4.1f} {lean:5.1f} {lift:5.2f} | {(com.z - right.z) * 100:+6.1f} (foot half-width "
          f"{R.FOOT_WIDTH / 2 * 100:.0f}) | {left_up * 100:5.1f} | {roll:+5.1f} | right leg off target / torque: {motors}")
    if rb["pelvis"].transform.position.y < W.FALL_HEIGHT or up.y < math.cos(math.radians(W.FALL_TILT_DEG)):
        print(f"fell at {t:.1f} s")
        break


