"""
Reference motions: what each training stage asks the joints to do, as a function of time.
The policy's actions are corrections on top of these, so from the first episode the robot already
roughly performs the motion and only has to learn to keep its balance while doing it.

All angles in degrees, human convention (flexion positive), in servo order (left leg, then right).
Roll joints are mirrored per side: a positive "lean" moves the weight toward the right foot.
"""
import math

import robot as R

# Standing pose, tested to hold still. With the shin vertical a bent knee tips the thigh (and the center
# of mass) back, and the ankle is only 5 cm from the heel, so the ankle leans the shin forward and the hip
# takes the rest: pelvis lean = ankle - knee + hip = 0.
DEFAULT_POSE = {"ankle_pitch": 7.0, "ankle_roll": 0.0, "knee_flex": 10.0, "hip_roll": 0.0, "hip_flex": 3.0,
                "hip_yaw": 0.0}

TASKS = ["stand", "shift", "one_leg", "march", "walk"]

KNEE_LIFT_DEG = 40.0         # knee flexion added to the swinging leg (stepping stages)
# One-leg stage, measured open loop (tests/probe_one_leg_reference.py): with the 12 deg shift the center of
# mass was still 5 cm from the standing foot's center (1 cm from its edge) when the lift began, the robot
# rolled onto the lifted foot and it never left the floor. 18 deg puts it over the foot's center (roll
# joints stop at 20). Knee 40 / hip 20 lifted the foot only ~4.5 cm in theory, ~1 cm after the pelvis sag.
ONE_LEG_SHIFT_DEG = 18.0
ONE_LEG_KNEE_LIFT_DEG = 60.0
ONE_LEG_HIP_LIFT_DEG = 30.0  # hip flexion added while holding a foot up (~9 cm of lift in theory)
# The standing hip-roll servo gives ~4 deg under the load of the body on one leg, the pelvis sags toward the
# lifted leg and the robot tips that way; the reference hikes that side of the pelvis by as much.
ONE_LEG_HIKE_DEG = 4.0
# Weight shift: both legs lean sideways together (ankle roll, undone at the hip to keep the pelvis level).
# Measured: 6 deg moves the center of mass ~3 cm; the feet are 12 cm apart at their inner edges.
SWAY_DEG = 10.0              # during walking (dynamic: the capture point moves further than the CoM)
SHIFT_DEG = 12.0             # full static shift over one foot (shift and one-leg stages)
SHIFT_PERIOD = 5.0           # s for a full left-right-left shift (was 3: slower gives time to settle)
# Each half of the one-leg cycle starts by standing still on both feet: the episode starts from a settled
# robot, every lift starts from a stable stance, and two-foot standing keeps being practiced in this stage.
# Durations in s, in order. Slow on purpose: time to settle at every step of the motion.
# Pause 2.5 s (was 1.5) and the weight comes back in 1.5 s (was 1.0): the first ~0.5 s of each pause is the
# weight-back motion dying out, so a short pause left little real standing still (still_stable stuck ~0.55).
ONE_LEG_PAUSE = 2.5          # standing still before each weight shift
ONE_LEG_LEAN = 1.2           # moving the weight over the standing foot
# Lift and lower slowly (were 0.6 s, fast enough to tip over before the balance point was found): the foot
# unloads a little at a time, the policy keeps finding the balance over the standing foot as it rises.
ONE_LEG_LIFT = 2.5           # lifting the other foot
ONE_LEG_HOLD = 2.0           # holding it up
ONE_LEG_LOWER = 1.5          # putting it down
ONE_LEG_BACK = 1.5           # moving the weight back to the middle
ONE_LEG_CYCLE = 2 * (ONE_LEG_PAUSE + ONE_LEG_LEAN + ONE_LEG_LIFT + ONE_LEG_HOLD + ONE_LEG_LOWER + ONE_LEG_BACK)
GAIT_PERIOD = 1.6            # s per full step cycle (left + right); was 1.2, slower = more stable
SWAY_LEAD = 0.08             # of a gait cycle: the weight moves before the other foot lifts
GAIT_START_SECONDS = 1.5     # the step grows from nothing over this long, like starting to walk from rest


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _leg(side, lean=0.0, knee_add=0.0, hip_add=0.0, hike=0.0):
    """One leg's targets: default pose + a sideways lean + extra knee/hip bend with the foot kept level.
    hike (standing leg): extra hip roll that lifts the pelvis on the other side, against its sag."""
    mirror = 1.0 if side == 0 else -1.0
    knee = DEFAULT_POSE["knee_flex"] + knee_add
    hip = DEFAULT_POSE["hip_flex"] + hip_add
    # Keep the foot parallel to the floor under an upright pelvis (foot lean = ankle - knee + hip).
    ankle = DEFAULT_POSE["ankle_pitch"] + knee_add - hip_add
    # lean is a world direction (positive = to the right), so each leg turns it into its own convention with
    # mirror; hike is already the standing leg's own amount, the same on both sides. It used to be
    # "+ mirror * hike": right stance hiked correctly, left stance 8 deg apart from it (tipping the wrong way).
    pose = {"ankle_pitch": ankle, "ankle_roll": mirror * lean, "knee_flex": knee,
            "hip_roll": -mirror * lean - hike, "hip_flex": hip, "hip_yaw": 0.0}
    return [pose[j] for j in R.JOINT_ORDER]


def hip_swing_amplitude(command_speed):
    return 3.0 + 25.0 * command_speed   # deg: longer steps for faster commands (0.1 m/s -> 5.5 deg)


def _gait_leg(side, phase, command_speed, scale):
    s = (phase + 0.5 * side) % 1.0
    amplitude = hip_swing_amplitude(command_speed) * scale
    if s < 0.5:                                # swing: hip from back to front, knee lifts and lowers
        u = s / 0.5
        hip_add = -amplitude * math.cos(math.pi * u)
        knee_add = KNEE_LIFT_DEG * scale * math.sin(math.pi * u)
    else:                                      # stance: hip pushes from front to back, knee near straight
        v = (s - 0.5) / 0.5
        hip_add = amplitude * math.cos(math.pi * v)
        knee_add = 0.0
    lean = SWAY_DEG * scale * math.sin(2 * math.pi * (phase + SWAY_LEAD))
    return _leg(side, lean, knee_add, hip_add)


def one_leg_schedule(t, first_leg=0, only_stance=None):
    """(lean, lifted side, lift amount 0..1) at time t of the one-leg stage; first_leg lifts first (0 left).
    only_stance (0 left / 1 right): stand on that foot in every half cycle (practising the weaker leg)."""
    half = ONE_LEG_CYCLE / 2
    u = t % ONE_LEG_CYCLE
    lifted = first_leg if u < half else 1 - first_leg
    if only_stance is not None:
        lifted = 1 - only_stance
    u = u % half - ONE_LEG_PAUSE
    if u < 0.0:
        return 0.0, lifted, 0.0                 # standing still on both feet
    direction = 1.0 if lifted == 0 else -1.0    # positive lean = weight to the right
    lifted_at = ONE_LEG_LEAN
    held_at = lifted_at + ONE_LEG_LIFT
    lower_at = held_at + ONE_LEG_HOLD
    back_at = lower_at + ONE_LEG_LOWER
    if u < lifted_at:
        lean, lift = _smooth(u / ONE_LEG_LEAN), 0.0
    elif u < held_at:
        lean, lift = 1.0, _smooth((u - lifted_at) / ONE_LEG_LIFT)
    elif u < lower_at:
        lean, lift = 1.0, 1.0                   # hold on one foot
    elif u < back_at:
        lean, lift = 1.0, 1.0 - _smooth((u - lower_at) / ONE_LEG_LOWER)
    else:
        lean, lift = 1.0 - _smooth((u - back_at) / ONE_LEG_BACK), 0.0
    return direction * ONE_LEG_SHIFT_DEG * lean, lifted, lift


def one_leg_still(t, settle=0.0):
    """True while the one-leg stage asks to stand still on both feet (the pause before each shift); with
    settle, only from that many seconds into the pause."""
    return settle <= t % (ONE_LEG_CYCLE / 2) < ONE_LEG_PAUSE


def reference_angles(task, t, phase, command_speed, first_leg=0, only_stance=None):
    """Targets for all servos. t: seconds into the episode; phase: gait clock in [0, 1)."""
    if task == "stand":
        return _leg(0) + _leg(1)
    if task == "shift":
        lean = SHIFT_DEG * math.sin(2 * math.pi * t / SHIFT_PERIOD)
        return _leg(0, lean) + _leg(1, lean)
    if task == "one_leg":
        lean, lifted, lift = one_leg_schedule(t, first_leg, only_stance)
        legs = []
        for side in (0, 1):
            up = lift if side == lifted else 0.0
            hike = ONE_LEG_HIKE_DEG * lift if side != lifted else 0.0
            legs += _leg(side, lean, ONE_LEG_KNEE_LIFT_DEG * up, ONE_LEG_HIP_LIFT_DEG * up, hike)
        return legs
    speed = 0.0 if task == "march" else command_speed
    scale = min(1.0, t / GAIT_START_SECONDS)
    return _gait_leg(0, phase, speed, scale) + _gait_leg(1, phase, speed, scale)


def gait_windows(phase):
    """(left should swing, right should swing) for the stepping stages."""
    return 0.1 < phase < 0.4, 0.6 < phase < 0.9
