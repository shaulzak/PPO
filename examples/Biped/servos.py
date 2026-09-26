"""
Servo models the biped can be built and trained with. Everything that depends on the servo
(physics, observations, current estimate) reads it from SERVO.

Pick the model with the BIPED_SERVO environment variable (or a whole build with BIPED_BUILD, below):
    set BIPED_SERVO=RDS3115MG_6V      (Windows cmd)
    $env:BIPED_SERVO = "RDS3115MG_6V" (PowerShell)
The model a checkpoint was trained with is stored in it; Run/Evaluate/--resume warn on a mismatch.

BIPED_BUILD picks a whole build at once (BUILDS below): "default" (SHOPPING_LIST.md: 10 x STS3250 + 2 x STS3095
at the hip roll, hip yaw), "v2" (12 x STS3250), "hybrid" (default without hip yaw) or "current" (the robot as it
is today: 10 x RDS3115MG, 1120 channels, no hip yaw, no battery). BIPED_SERVO / BIPED_HIP_YAW /
BIPED_LEG_CHANNEL / BIPED_BATTERY still override single parts of it.
"""
import os
from dataclasses import dataclass

KG_CM = 0.0980665   # N*m per kg*cm


@dataclass(frozen=True)
class ServoSpec:
    name: str
    stall_torque: float          # N*m, the most it can push
    rated_torque: float          # N*m it can hold continuously without overheating
    max_speed: float             # deg/s, no load
    mass: float                  # kg, the servo alone (brackets are added in robot.py)
    stall_current: float         # A at stall torque (current is estimated from the torque used)
    idle_current: float          # A when holding with no load
    voltage: float               # V it is powered at
    position_feedback: bool      # reports its real angle back (the policy then observes it)
    armature: float              # kg*m^2, gearbox inertia felt at the output: rotor inertia * ratio^2 (estimate)
    interface: str               # how the controller talks to it
    size_mm: tuple               # body size, for the mechanical build


SERVOS = {
    # What the robot has now. PWM only: the controller never learns the real angle.
    # Rated torque not published; a quarter of stall, like the STS3250.
    "RDS3115MG_6V": ServoSpec(
        name="RDS3115MG_6V", stall_torque=15 * KG_CM, rated_torque=15 * KG_CM / 4, max_speed=60 / 0.14,
        mass=0.060, stall_current=2.5, idle_current=0.1, voltage=6.0, position_feedback=False,
        armature=0.005, interface="PWM through PCA9685 (50 Hz)", size_mm=(40, 20, 40.5)),
    # On the shopping list. Serial bus with a 12-bit magnetic encoder, 1:345 gearbox. Electrical figures
    # from Feetech's 50 kg*cm TTL spec sheet (HLS3950M/STS3250, 2024-04-25): stall 2.4 A, idle 24 mA,
    # rated 12.5 kg*cm at 0.6 A; overload protection cuts in above 80% of stall torque for 2 s.
    "STS3250_12V": ServoSpec(
        name="STS3250_12V", stall_torque=50 * KG_CM, rated_torque=12.5 * KG_CM, max_speed=60 / 0.133,
        mass=0.0745, stall_current=2.4, idle_current=0.024, voltage=12.0, position_feedback=True,
        armature=0.01, interface="TTL serial bus (half duplex)", size_mm=(45.2, 24.7, 35)),
    # Hip-roll upgrade on the shopping list: same bus and protocol, ~2x the torque, heavier and slower
    # (31 rpm). Rated torque assumed a quarter of stall and the armature scaled with the gearbox load.
    "STS3095_12V": ServoSpec(
        name="STS3095_12V", stall_torque=95 * KG_CM, rated_torque=95 * KG_CM / 4, max_speed=31 * 6,
        mass=0.1945, stall_current=9.8, idle_current=0.05, voltage=12.0, position_feedback=True,
        armature=0.02, interface="TTL serial bus (half duplex)", size_mm=(30, 65, 48)),
}

# BIPED_HIP_ROLL_SERVO: a different model at the two hip-roll joints (empty = BIPED_SERVO everywhere).
BUILDS = {
    # The default: SHOPPING_LIST.md's recommendation, 10 x STS3250 + 2 x STS3095 at the hip roll, with hip yaw.
    "default": {"BIPED_SERVO": "STS3250_12V", "BIPED_HIP_YAW": "1", "BIPED_LEG_CHANNEL": "1121", "BIPED_BATTERY": "1",
                "BIPED_HIP_ROLL_SERVO": "STS3095_12V"},
    # 12 x STS3250 (what the models up to 2026-09-26 were trained with).
    "v2": {"BIPED_SERVO": "STS3250_12V", "BIPED_HIP_YAW": "1", "BIPED_LEG_CHANNEL": "1121", "BIPED_BATTERY": "1",
           "BIPED_HIP_ROLL_SERVO": ""},
    # 8 x STS3250 + 2 x STS3095 at the hip roll, no hip yaw.
    "hybrid": {"BIPED_SERVO": "STS3250_12V", "BIPED_HIP_YAW": "0", "BIPED_LEG_CHANNEL": "1121", "BIPED_BATTERY": "1",
               "BIPED_HIP_ROLL_SERVO": "STS3095_12V"},
    # RDS3115MG can't hold the robot on one leg: tests/test_one_leg_strength.py.
    "current": {"BIPED_SERVO": "RDS3115MG_6V", "BIPED_HIP_YAW": "0", "BIPED_LEG_CHANNEL": "1120", "BIPED_BATTERY": "0",
                "BIPED_HIP_ROLL_SERVO": ""},
}
BUILD = os.environ.get("BIPED_BUILD", "default")
if BUILD not in BUILDS:
    raise ValueError(f"Unknown BIPED_BUILD={BUILD!r}; choose one of {sorted(BUILDS)}")


def build_setting(name):
    """A build option: its environment variable if set, else the BIPED_BUILD preset's value."""
    return os.environ.get(name, BUILDS[BUILD][name])


SERVO_MODEL = build_setting("BIPED_SERVO")
if SERVO_MODEL not in SERVOS:
    raise ValueError(f"Unknown BIPED_SERVO={SERVO_MODEL!r}; choose one of {sorted(SERVOS)}")
SERVO = SERVOS[SERVO_MODEL]
HIP_ROLL_MODEL = build_setting("BIPED_HIP_ROLL_SERVO") or SERVO_MODEL
if HIP_ROLL_MODEL not in SERVOS:
    raise ValueError(f"Unknown BIPED_HIP_ROLL_SERVO={HIP_ROLL_MODEL!r}; choose one of {sorted(SERVOS)}")
if SERVOS[HIP_ROLL_MODEL].position_feedback != SERVO.position_feedback:
    raise ValueError("BIPED_HIP_ROLL_SERVO must have position feedback like BIPED_SERVO (the observation needs it)")


def servo_for(joint):
    """The servo model at a joint (hip_roll may differ, the rest use SERVO)."""
    return SERVOS[HIP_ROLL_MODEL] if joint == "hip_roll" else SERVO


def servos_label():
    """e.g. "STS3250_12V" or "STS3250_12V + STS3095_12V at hip roll"."""
    return SERVO.name + (f" + {HIP_ROLL_MODEL} at hip roll" if HIP_ROLL_MODEL != SERVO_MODEL else "")
