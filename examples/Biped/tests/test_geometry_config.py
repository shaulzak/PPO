"""
Stability geometry (support area, capture point), reference motions and the robot configuration.
    python tests/test_geometry_config.py
"""
import math

from _common import check, finish

import robot as R
import motions as M
from servos import SERVO, SERVOS
from walk_metrics import support_margin, capture_point, convex_hull
import WalkAgent as W

print(f"test_geometry_config ({SERVO.name}, {R.NUM_SERVOS} servos)")

square = [(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]
check("support margin: center of a 10 cm square is 5 cm inside", abs(support_margin((0.05, 0.05), square) - 0.05) < 1e-9)
check("support margin: 2 cm outside is -2 cm", abs(support_margin((0.12, 0.05), square) + 0.02) < 1e-9)
check("support margin: no contacts is -0.1", support_margin((0, 0), []) == -0.1)
check("convex hull drops inner points", len(convex_hull(square + [(0.05, 0.05)])) == 4)
cp = capture_point((0.0, 0.0), (0.3, 0.0), 0.4)
check("capture point: 0.3 m/s at 0.4 m height lands ~6 cm ahead", abs(cp[0] - 0.3 * math.sqrt(0.4 / 9.81)) < 1e-9,
      f"{cp[0] * 100:.1f} cm")

# every reference target stays inside the joint limits (otherwise it is silently clipped)
worst = {}
for task in M.TASKS:
    for k in range(0, 1200):
        t = k * 0.01
        angles = M.reference_angles(task, t, (t / M.GAIT_PERIOD) % 1.0, 0.12)
        for i, a in enumerate(angles):
            s_low, s_high = R.JOINT_LIMITS[R.JOINT_ORDER[i % R.JOINTS_PER_LEG]]
            outside = max(s_low - a, a - s_high, 0.0)
            worst[task] = max(worst.get(task, 0.0), outside)
check("reference motions stay inside the joint limits", max(worst.values()) < 1e-6,
      ", ".join(f"{k} {v:.1f} deg over" for k, v in worst.items() if v > 0) or "all inside")

# configuration
check("observation size matches the configuration", W.OBS_DIM ==
      3 + 3 + 3 * R.NUM_SERVOS + (R.NUM_SERVOS if SERVO.position_feedback else 0) + 8 + 2 + len(M.TASKS) + 1 + 2)
check("12 servos with hip yaw, 10 without", R.NUM_SERVOS == (12 if R.HIP_YAW else 10))
check("every servo spec has a rated torque below its stall torque",
      all(0 < s.rated_torque < s.stall_torque for s in SERVOS.values()))
root, robot = R.build_robot()
mass = sum(b.Rigidbody.mass for b in [robot["pelvis"]] + list(robot["left"].values()) + list(robot["right"].values()))
check("robot mass is plausible (2-4.5 kg; ~3.9 with 12 heavy STS3095)", 2.0 < mass < 4.5, f"{mass:.2f} kg")

finish()
