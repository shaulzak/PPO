"""
Are the servos strong enough to stand on one leg? Independent of training: no policy can hold a pose the
motors can't.
1. Statics: the one-leg hold pose (left foot up), leaned until the center of mass is right above the standing
   ankle (the pose that needs the least ankle torque), posed in zero gravity with the pelvis held. The torque
   gravity puts on every standing-leg joint is the weight of everything on the pelvis side of it times its
   lever, with gravity along the standing foot's "down".
2. Simulation: the same motion with gravity and the standing foot glued to the floor (so balance can't fail,
   only strength). The joints must hold their targets if statics says the torque is below stall, and give way
   if it is above.
Prints a verdict for the configuration; checks that statics and simulation agree.
    python tests/test_one_leg_strength.py
    BIPED_BUILD=current python tests/test_one_leg_strength.py      (the robot of today)
"""
from _common import check, finish, make_world

from bereshit import Vector3
import robot as R
import motions as M
import WalkAgent as W
from servos import servo_for, servos_label

print(f"test_one_leg_strength ({servos_label()}, {R.NUM_SERVOS} servos, channel {R.LEG_CHANNEL}, "
      f"battery {'yes' if R.BATTERY else 'no'})")

STANCE = 1                                     # the right foot stands, the left one is lifted
CHAIN = ["yaw_block", "hip_block", "thigh", "shin", "ankle_block", "foot"]   # pelvis -> foot
HOST = {"hip_yaw": "yaw_block", "hip_flex": "hip_block", "hip_roll": "thigh", "knee_flex": "shin",
        "ankle_roll": "ankle_block", "ankle_pitch": "foot"}   # the body each servo turns


def copy(v):
    return Vector3(v.x, v.y, v.z)   # transform.position is a live reference into the engine


def pose(lean, lift, ankle_pitch_add):
    """One-leg targets: lean (deg, + = weight to the right), lift 0..1 of the left foot, extra pitch on the
    standing ankle (tips the whole robot forward/back to put the center of mass over the ankle)."""
    angles = (M._leg(0, lean, M.ONE_LEG_KNEE_LIFT_DEG * lift, M.ONE_LEG_HIP_LIFT_DEG * lift)
              + M._leg(1, lean, 0.0, 0.0, M.ONE_LEG_HIKE_DEG * lift))
    angles[R.servo_index("ankle_pitch", STANCE)] += ankle_pitch_add
    return angles


class Scene:
    def __init__(self, gravity, glue_foot):
        self.root, self.rb = R.build_robot(lift=0.3 if not glue_foot else 0.002)
        self.bodies = [self.rb["pelvis"]] + list(self.rb["left"].values()) + list(self.rb["right"].values())
        self.leg = self.rb["right"]
        self.start = {id(b): copy(b.transform.position) for b in self.bodies}
        objects = [self.root]
        if glue_foot:
            self.leg["foot"].Rigidbody.isKinematic = True
            objects.append(R.build_floor())
        else:
            self.rb["pelvis"].Rigidbody.isKinematic = True
        self.world = make_world(objects, gravity=gravity, tick=W.TICK, epochs=W.PHYSICS_EPOCHS)

    def servo(self, joint):
        return self.rb["servos"][R.servo_index(joint, STANCE)]

    def run(self, angles, seconds):
        for s, a in zip(self.rb["servos"], angles):
            s.command(a)
        for _ in range(round(seconds / W.TICK)):
            self.world.update(False)

    def anchor(self, joint):
        """World position of a standing-leg joint axis (the anchor moves with the body the servo turns)."""
        body = self.leg[HOST[joint]]
        return body.transform.position + R.local_to_world(body.transform.quaternion,
                                                          self.servo(joint).anchor - self.start[id(body)])

    def foot_frame(self):
        q = self.leg["foot"].transform.quaternion
        return (R.local_to_world(q, Vector3(1, 0, 0)), R.local_to_world(q, Vector3(0, 0, 1)),
                R.local_to_world(q, Vector3(0, -1, 0)))

    def com(self):
        mass = sum(b.Rigidbody.mass for b in self.bodies)
        return sum((b.transform.position * b.Rigidbody.mass for b in self.bodies), Vector3(0, 0, 0)) * (1 / mass)

    def gravity_torques(self):
        """N*m each standing-leg joint must give to hold the pose still, gravity along the foot's down."""
        g = self.foot_frame()[2] * 9.81
        torques = {}
        for joint in R.JOINT_ORDER:
            foot_side = [self.leg[n] for n in CHAIN[CHAIN.index(HOST[joint]):]]
            carried = [b for b in self.bodies if all(b is not f for f in foot_side)]
            p = self.anchor(joint)
            moment = sum(((b.transform.position - p).cross(g * b.Rigidbody.mass) for b in carried), Vector3(0, 0, 0))
            torques[joint] = abs(moment.dot(self.servo(joint).axis_world()))
        return torques


# 1. statics
hang = Scene(gravity=(0, 0, 0), glue_foot=False)
lean, pitch = M.ONE_LEG_SHIFT_DEG, 0.0
for _ in range(20):
    hang.run(pose(lean, 1.0, pitch), 1.5)
    forward, side, _ = hang.foot_frame()
    d = hang.com() - hang.anchor("ankle_pitch")
    off_fwd, off_side = d.dot(forward), d.dot(side)   # side: + = toward the lifted (left) leg
    if abs(off_fwd) < 0.002 and abs(off_side) < 0.002:
        break
    lean += 100 * off_side / 0.6 * 0.8       # measured: ~0.6 cm of center of mass per degree of lean
    pitch -= 100 * off_fwd / 0.9 * 0.8       # ~0.9 cm per degree of ankle pitch
check("balanced one-leg pose found (center of mass within 2 mm of the standing ankle)",
      abs(off_fwd) < 0.002 and abs(off_side) < 0.002,
      f"lean {lean:.1f} deg, standing ankle pitch {pitch:+.1f} deg, off {off_fwd * 100:+.1f} / {off_side * 100:+.1f} cm")
need = hang.gravity_torques()
print(f"  holding still on the right foot (robot {sum(b.Rigidbody.mass for b in hang.bodies):.2f} kg):")
for joint, tau in need.items():
    spec = servo_for(joint)
    print(f"    {joint:<11} {tau:4.2f} N*m = {tau / spec.stall_torque:4.0%} of stall ({spec.stall_torque:.2f}), "
          f"{tau / spec.rated_torque:5.0%} of rated ({spec.rated_torque:.2f}, {spec.name})")
worst = max(need, key=lambda j: need[j] / servo_for(j).stall_torque)
worst_spec = servo_for(worst)
strong_enough = need[worst] < worst_spec.stall_torque

# 2. simulation: lean, lift, hold, with the standing foot glued down
stand = Scene(gravity=(0, -9.8, 0), glue_foot=True)
pelvis = stand.rb["pelvis"]
stand.run(pose(0.0, 0.0, 0.0), 1.0)
start_height = pelvis.transform.position.y
steps = 20
for k in range(1, steps + 1):                  # lean over 1.2 s, then lift over 2.5 s (as in the stage)
    stand.run(pose(lean * min(1.0, k / 6), max(0.0, (k - 6) / (steps - 6)), pitch * max(0.0, (k - 6) / (steps - 6))),
              (1.2 / 6) if k <= 6 else 2.5 / (steps - 6))
worst_error = {j: 0.0 for j in R.JOINT_ORDER}
lowest = pelvis.transform.position.y
for _ in range(30):                            # hold 3 s
    stand.run(pose(lean, 1.0, pitch), 0.1)
    for j in R.JOINT_ORDER:
        s = stand.servo(j)
        worst_error[j] = max(worst_error[j], abs(s.angle() - s.command_deg))
    lowest = min(lowest, pelvis.transform.position.y)
left_foot_up = stand.rb["left"]["foot"].transform.position.y - R.FOOT_THICKNESS / 2 - R.FLOOR_TOP   # its center
print("  simulated hold (standing foot glued): standing-leg joints off target by up to " +
      ", ".join(f"{j} {e:.1f}" for j, e in worst_error.items()) + " deg")
print(f"  pelvis {(start_height - lowest) * 100:.1f} cm lower than standing, left foot {left_foot_up * 100:.1f} cm up")
# A servo holding a load sits a few degrees off its target (~4 at the hip roll: the reason for ONE_LEG_HIKE_DEG).
held = max(worst_error.values()) < 6.0 and left_foot_up > 0.03
if strong_enough:
    check("statics says strong enough, and the simulated joints hold (< 6 deg, foot > 3 cm up)", held)
else:
    check("statics says too weak, and the simulated joints give way (> 6 deg or the foot comes down)", not held)

print(f"  VERDICT {servos_label()}: the hardest joint is {worst} ({worst_spec.name}), {need[worst]:.2f} N*m = "
      f"{need[worst] / worst_spec.stall_torque:.0%} of stall, {need[worst] / worst_spec.rated_torque:.0%} of rated -> "
      + ("CAN hold one leg (by strength)" if strong_enough else "CANNOT stand on one leg (too weak, whatever the policy)"))
finish()
