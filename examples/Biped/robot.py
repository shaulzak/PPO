"""
The real biped: 1120 U-channels, aluminum feet, and the servo model chosen in servos.py.

Units: meters, kilograms, seconds; joint angles in degrees.
Axes: x = forward, y = up, z = sideways (left leg at +z).

Engine conventions (measured, see the notes in HobbyServo):
- q.conjugate().rotate(v) maps a body-local vector to world; q.rotate(v) maps world to body-local.
"""
import math

from bereshit import GameObject, Vector3, BoxCollider, Rigidbody, HingeJoint, Component

import os

from servos import SERVO, SERVO_MODEL, build_setting, servo_for

# --- build configuration (BIPED_BUILD in servos.py; by default SHOPPING_LIST.md's recommendation) ---
# Hip yaw: a 6th servo per leg turning the leg about the vertical axis, between the pelvis and the hip.
HIP_YAW = build_setting("BIPED_HIP_YAW") == "1"
HIP_YAW_STACK = 0.045         # m the yaw servo adds between the hip pitch axis and the pelvis (estimate: re-measure)
# Shin/thigh channel: goBILDA 1121 low-side (66 g) or 1120 (128 g), same hole pattern.
LEG_CHANNEL = build_setting("BIPED_LEG_CHANNEL")
CHANNEL_MASSES = {"1120": 0.128, "1121": 0.066}
if LEG_CHANNEL not in CHANNEL_MASSES:
    raise ValueError(f"Unknown BIPED_LEG_CHANNEL={LEG_CHANNEL!r}; choose one of {sorted(CHANNEL_MASSES)}")

# --- measured geometry (joint axis heights are above the top of the foot plate) ---
HIP_SPACING = 0.2443          # between the centers of the two hip servos
FOOT_LENGTH, FOOT_WIDTH = 0.25, 0.12
FOOT_THICKNESS = 0.01         # real plate is 1 mm; 10 mm keeps collisions stable, mass is the real one
ANKLE_FROM_HEEL = 0.05
ANKLE_PITCH_H = 0.0339
ANKLE_ROLL_H = 0.0678
ANKLE_ROLL_TO_KNEE = 0.2527
KNEE_TO_HIP_ROLL = 0.2527
HIP_ROLL_TO_HIP_PITCH = 0.0769
HIP_PITCH_TO_PELVIS_BOTTOM = 0.0337
CHANNEL = 0.048               # 1120 U-channel cross-section
PELVIS_LENGTH = 0.312
BLOCK_SIZE = 0.04             # collision box of the two servos between a pair of axes
# The engine doesn't skip collisions between jointed bodies, so segment boxes stop short of each joint
# (half the channel width + margin) to let neighbours rotate without touching.
JOINT_GAP = 0.03

# --- servo (chosen in servos.py) ---
SERVO_MAX_TORQUE = SERVO.stall_torque
SERVO_MAX_SPEED = SERVO.max_speed

# --- masses (version 2 build: ~2.4 kg). Each rigid segment is one body. ---
SERVO_MASS = SERVO.mass + 0.03   # + brackets and horn


def servo_mass(joint):
    return servo_for(joint).mass + 0.03   # + brackets and horn


FOOT_MASS = 0.10              # 1 mm plate (81 g) + rubber sole and FSRs
LEG_WIRING_MASS = 0.025       # per channel: cables, sensor wiring
ANKLE_BLOCK_MASS = servo_mass("ankle_pitch") + servo_mass("ankle_roll")
SHIN_MASS = CHANNEL_MASSES[LEG_CHANNEL] + LEG_WIRING_MASS + servo_mass("knee_flex")   # + the knee servo
THIGH_MASS = CHANNEL_MASSES[LEG_CHANNEL] + LEG_WIRING_MASS
HIP_BLOCK_MASS = servo_mass("hip_flex") + servo_mass("hip_roll")
YAW_BLOCK_MASS = servo_mass("hip_yaw")
# 1120-0012-0312 (206 g) + ESP32, bus adapter, IMU, switch, fuse, wiring (~150 g), + the LiPo mounted in
# the middle of the pelvis: there it sits 12 cm from a standing hip instead of 24 cm on the other leg,
# and it isn't swung with every step.
# The robot of today has no battery (bench supply through a cable).
BATTERY = build_setting("BIPED_BATTERY") == "1"
BATTERY_MASS = 0.24 if BATTERY else 0.0   # 3S 3000 mAh
PELVIS_MASS = 0.206 + 0.15 + BATTERY_MASS

# Reflected gearbox inertia ("armature", per servo model in servos.py). Added to the ankle/hip blocks,
# which are otherwise so light (~3e-5 kg*m^2) that the joint solver can't pass the servo torque through
# them to the rest of the leg.
SERVO_ARMATURE = SERVO.armature
# per block: the servos it holds (a block with a heavier servo gets that one's armature)
BLOCK_SERVOS = {"yaw_block": ("hip_yaw",), "hip_block": ("hip_flex", "hip_roll"),
                "ankle_block": ("ankle_pitch", "ankle_roll")}

# (low, high) in degrees, in human terms. The FLEX_SIGNs map "flexion is positive" to the measured
# angle; found by commanding each joint in a hanging robot (knee: shin swings back; hip: foot forward).
JOINT_LIMITS = {
    # Up to 50 forward: to keep a swinging foot level, the ankle must undo the knee bend (ankle = knee - hip,
    # ~47 deg in mid-swing). At 30 the toe hung ~6 cm down and scraped the floor even on a perfect step.
    "ankle_pitch": (-30.0, 50.0),
    # Roll: +-30 (human hip ~30-45, ankle ~20-30). The one-leg stage needs 18 deg of lean + a 4 deg hip hike;
    # 20 was an unmeasured guess. CHECK ON THE REAL ROBOT that nothing collides at 30.
    "ankle_roll": (-30.0, 30.0),
    "knee_flex": (0.0, 110.0),      # bends only one way, like a human knee
    "hip_roll": (-30.0, 30.0),
    "hip_flex": (-30.0, 80.0),      # 30 back, 80 forward
    "hip_yaw": (-30.0, 30.0),       # toes in / out
}
KNEE_FLEX_SIGN = -1.0
HIP_FLEX_SIGN = 1.0

JOINT_ORDER = ["ankle_pitch", "ankle_roll", "knee_flex", "hip_roll", "hip_flex"] + (["hip_yaw"] if HIP_YAW else [])
JOINTS_PER_LEG = len(JOINT_ORDER)
NUM_SERVOS = 2 * JOINTS_PER_LEG


def servo_index(joint, side):
    """Index into robot['servos'] (left leg first). side: 0 left, 1 right."""
    return side * JOINTS_PER_LEG + JOINT_ORDER.index(joint)


FLOOR_TOP = 0.0
PITCH_AXIS = Vector3(0, 0, 1)
ROLL_AXIS = Vector3(1, 0, 0)
YAW_AXIS = Vector3(0, 1, 0)


def _cap(value, limit):
    return min(max(value, -limit), limit)


def local_to_world(q, v):
    return q.conjugate().rotate(v)


def world_to_local(q, v):
    return q.rotate(v)


class Floor(Component):
    """Marker for the ground, so collision callbacks can recognize it."""


class HobbyServo(Component):
    """
    A position servo like the RDS3115MG, built on the HingeJoint motor: it turns toward
    the commanded angle at up to max_speed deg/s (slowing down within max_speed / gain
    degrees of it) and never pushes harder than max_torque. The motor is solved with the
    joint constraints, so the reaction on the mount and the drag of the rest of the leg
    are handled by the engine.

    The angle is measured from world directions of a reference vector on each
    body, so it doesn't depend on the engine's quaternion conventions.
    """

    # Set by the servo test: sign of the motor speed that increases the measured angle.
    MOTOR_SIGN = 1.0
    # Position correction strength of the hinge (engine default 0.2). With 50 solver iterations,
    # 0.5 keeps the light ankle/hip blocks aligned (< 0.6 deg leak while standing) at ~16x real time.
    JOINT_BETA = 0.5

    def __init__(self, mount, axis, anchor, limits, max_torque=SERVO_MAX_TORQUE, max_speed=SERVO_MAX_SPEED,
                 gain=20.0, sign=1.0):
        super(HobbyServo, self).__init__()
        self.mount = mount
        self.anchor = anchor
        self.axis_local = axis.normalized()
        self.low, self.high = limits
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.gain = gain  # 1/s: commanded speed = gain * error, up to max_speed
        # sign maps the joint's human meaning (e.g. flexion) to the measured angle.
        self.sign = sign
        # A vector perpendicular to the axis, used to measure the angle.
        self.reference_local = Vector3(0, 1, 0) if abs(self.axis_local.y) < 0.9 else Vector3(1, 0, 0)
        self.command_deg = 0.0

    def attach(self, parent):
        self.body = parent
        self.joint = HingeJoint(self.mount, self.axis_local, self.anchor, HobbyServo.JOINT_BETA)
        self.joint.motor_enabled = True
        self.joint.max_motor_torque = self.max_torque
        parent.add_component(self.joint)

    def axis_world(self):
        return local_to_world(self.mount.transform.quaternion, self.axis_local).normalized()

    def angle(self):
        """Joint angle in degrees, in the joint's human convention (flexion positive, etc.)."""
        return self.sign * self._raw_angle()

    def _raw_angle(self):
        axis = self.axis_world()
        r_mount = local_to_world(self.mount.transform.quaternion, self.reference_local)
        r_body = local_to_world(self.body.transform.quaternion, self.reference_local)
        return math.degrees(math.atan2(r_mount.cross(r_body).dot(axis), r_mount.dot(r_body)))

    def command(self, angle_deg):
        """Target angle in the joint's human convention; clamped to the joint limits."""
        self.command_deg = min(max(angle_deg, self.low), self.high)

    def ResetToDefault(self):
        self.command_deg = 0.0

    def PhysicsUpdate(self, dt):
        error = self.sign * self.command_deg - self._raw_angle()
        speed_deg = _cap(self.gain * error, self.max_speed)
        self.joint.motor_speed = HobbyServo.MOTOR_SIGN * math.radians(speed_deg)


def _spec(joint):
    """HobbyServo arguments of the servo model at a joint."""
    return {"max_torque": servo_for(joint).stall_torque, "max_speed": servo_for(joint).max_speed}


def _body(position, size, mass, name, *components, friction=0.6, restitution=0.0):
    return GameObject(position=position, size=size, name=name).add_component(
        BoxCollider(), Rigidbody(mass=mass, friction_coefficient=friction, restitution=restitution), *components)


def axis_heights(lift=0.002):
    """World heights of the joint axes and the pelvis center."""
    foot_top = FOOT_THICKNESS + lift
    h = {"foot_top": foot_top,
         "ankle_pitch": foot_top + ANKLE_PITCH_H,
         "ankle_roll": foot_top + ANKLE_ROLL_H}
    h["knee"] = h["ankle_roll"] + ANKLE_ROLL_TO_KNEE
    h["hip_roll"] = h["knee"] + KNEE_TO_HIP_ROLL
    h["hip_pitch"] = h["hip_roll"] + HIP_ROLL_TO_HIP_PITCH
    h["pelvis_bottom"] = h["hip_pitch"] + HIP_PITCH_TO_PELVIS_BOTTOM + (HIP_YAW_STACK if HIP_YAW else 0.0)
    h["hip_yaw"] = h["pelvis_bottom"]          # the yaw servo turns the leg about a vertical axis under the pelvis
    h["pelvis"] = h["pelvis_bottom"] + CHANNEL / 2
    return h


def build_leg(side, pelvis, lift=0.002):
    """
    One leg hanging from the pelvis. side = +1 (left, +z) or -1 (right, -z).
    Every segment is one rigid body; each hosts the servo that connects it to the segment above.
    Returns (leg object, parts dict, servos in JOINT_ORDER).
    """
    z = side * HIP_SPACING / 2
    h = axis_heights(lift)
    block = Vector3(BLOCK_SIZE, BLOCK_SIZE / 2, BLOCK_SIZE)

    def anchor(name):
        return Vector3(0, h[name], z)

    def channel_between(bottom, top):
        return Vector3(0, (h[bottom] + h[top]) / 2, z), Vector3(CHANNEL, h[top] - h[bottom] - 2 * JOINT_GAP, CHANNEL)

    blocks = []
    servos = {}
    above = pelvis
    if HIP_YAW:
        servos["hip_yaw"] = HobbyServo(pelvis, YAW_AXIS, anchor("hip_yaw"), JOINT_LIMITS["hip_yaw"], sign=side,
                                       **_spec("hip_yaw"))
        yaw_block = _body(Vector3(0, (h["hip_pitch"] + h["pelvis_bottom"]) / 2 + 0.005, z), block, YAW_BLOCK_MASS,
                          "yaw_block", servos["hip_yaw"])
        blocks.append(yaw_block)
        above = yaw_block
    hip_flex = HobbyServo(above, PITCH_AXIS, anchor("hip_pitch"), JOINT_LIMITS["hip_flex"], sign=HIP_FLEX_SIGN,
                          **_spec("hip_flex"))
    hip_block = _body(Vector3(0, (h["hip_roll"] + h["hip_pitch"]) / 2, z), block, HIP_BLOCK_MASS, "hip_block",
                      hip_flex)
    hip_roll = HobbyServo(hip_block, ROLL_AXIS, anchor("hip_roll"), JOINT_LIMITS["hip_roll"], sign=side, **_spec("hip_roll"))
    position, size = channel_between("knee", "hip_roll")
    thigh = _body(position, size, THIGH_MASS, "thigh", hip_roll)
    knee = HobbyServo(thigh, PITCH_AXIS, anchor("knee"), JOINT_LIMITS["knee_flex"], sign=KNEE_FLEX_SIGN,
                      **_spec("knee_flex"))
    position, size = channel_between("ankle_roll", "knee")
    shin = _body(position, size, SHIN_MASS, "shin", knee)
    ankle_roll = HobbyServo(shin, ROLL_AXIS, anchor("ankle_roll"), JOINT_LIMITS["ankle_roll"], sign=side,
                            **_spec("ankle_roll"))
    ankle_block = _body(Vector3(0, (h["ankle_pitch"] + h["ankle_roll"]) / 2, z), block, ANKLE_BLOCK_MASS,
                        "ankle_block", ankle_roll)
    ankle_pitch = HobbyServo(ankle_block, PITCH_AXIS, anchor("ankle_pitch"), JOINT_LIMITS["ankle_pitch"],
                             **_spec("ankle_pitch"))
    foot = _body(Vector3(FOOT_LENGTH / 2 - ANKLE_FROM_HEEL, lift + FOOT_THICKNESS / 2, z),
                 Vector3(FOOT_LENGTH, FOOT_THICKNESS, FOOT_WIDTH), FOOT_MASS, "foot", ankle_pitch)

    blocks += [hip_block, ankle_block]
    for blk, name in zip(blocks, (["yaw_block"] if HIP_YAW else []) + ["hip_block", "ankle_block"]):
        armature = max(servo_for(j).armature for j in BLOCK_SERVOS[name])
        i = blk.Rigidbody.inertia
        blk.Rigidbody.inertia = Vector3(i.x + armature, i.y + armature, i.z + armature)

    parts = {"hip_block": hip_block, "thigh": thigh, "shin": shin, "ankle_block": ankle_block, "foot": foot}
    if HIP_YAW:
        parts["yaw_block"] = blocks[0]
    servos.update({"ankle_pitch": ankle_pitch, "ankle_roll": ankle_roll, "knee_flex": knee,
                   "hip_roll": hip_roll, "hip_flex": hip_flex})
    leg = GameObject(size=Vector3(), children=list(parts.values()), name="leg_left" if side > 0 else "leg_right")
    return leg, parts, [servos[j] for j in JOINT_ORDER]


def build_robot(lift=0.002):
    """Returns (root object, robot dict with 'pelvis', 'left'/'right' parts and all NUM_SERVOS 'servos')."""
    h = axis_heights(lift)
    pelvis = _body(Vector3(0, h["pelvis"], 0), Vector3(CHANNEL, CHANNEL, PELVIS_LENGTH), PELVIS_MASS, "pelvis")
    leg_l, left, servos_l = build_leg(+1, pelvis, lift)
    leg_r, right, servos_r = build_leg(-1, pelvis, lift)
    root = GameObject(size=Vector3(), children=[pelvis, leg_l, leg_r], name="biped")
    robot = {"pelvis": pelvis, "left": left, "right": right, "servos": servos_l + servos_r}
    return root, robot


def build_floor():
    return GameObject(position=Vector3(0, FLOOR_TOP - 0.5, 0), size=Vector3(40, 1, 40), name="floor").add_component(
        BoxCollider(), Rigidbody(isKinematic=True, friction_coefficient=0.6, restitution=0.0), Floor())
