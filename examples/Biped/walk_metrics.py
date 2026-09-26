"""
Stability geometry and the training report.

Stability:
- support area: convex hull of the foot corners touching the floor (both feet, or the standing foot)
- capture point: where the center of mass would come to rest if the robot stopped stepping,
  CoM + velocity * sqrt(height / g). The robot can keep its balance without taking a step only while
  the capture point is inside the support area; unlike the plain CoM it also counts how fast the body
  is heading for the edge.
"""
import math

G = 9.81


def convex_hull(points):
    """Monotone chain; points as (x, z) tuples, returned counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _segment_distance(p, a, b):
    ax, az = b[0] - a[0], b[1] - a[1]
    length2 = ax * ax + az * az
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * az) / length2))
    return math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * az))


def support_margin(point, contact_points):
    """
    Signed distance (m) from a ground point to the support area (convex hull of the feet's contacts):
    positive inside (how far from the nearest edge), negative outside. No contacts -> -0.1.
    """
    hull = convex_hull(contact_points)
    if not hull:
        return -0.1
    if len(hull) == 1:
        return -math.hypot(point[0] - hull[0][0], point[1] - hull[0][1])
    edges = list(zip(hull, hull[1:] + hull[:1])) if len(hull) > 2 else [(hull[0], hull[1])]
    distance = min(_segment_distance(point, a, b) for a, b in edges)
    inside = len(hull) > 2 and all(
        (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0 for a, b in edges)
    return distance if inside else -distance


def capture_point(com_xz, com_velocity_xz, com_height):
    """Instantaneous capture point on the floor (x, z)."""
    t = math.sqrt(max(com_height, 1e-3) / G)
    return com_xz[0] + com_velocity_xz[0] * t, com_xz[1] + com_velocity_xz[1] * t


class WalkStats:
    """Averages of what the robot does between two reports, to see *how* it earns its reward."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.parts = {}
        self.metrics = {}
        self.events = {}
        self.stage_steps = {}
        # One-foot stretches: how long the robot carries itself on a single leg (human walking ~0.4 s).
        self.single_steps = {"L": 0, "R": 0}
        self.single_runs = {"L": 0, "R": 0}
        self._single_side = None
        self.touchdowns = 0
        self.step_lengths = []
        self.impacts = []
        # One-leg stage, per standing foot: sums of the hold measures, and hold steps.
        self.holds = {"L": {}, "R": {}}
        self.hold_steps = {"L": 0, "R": 0}
        # One-leg stage: how the robot stands as each weight shift begins, the episode's first lift | later ones.
        self.lift_starts = {"first": {}, "later": {}}
        self.lift_start_count = {"first": 0, "later": 0}

    def record_lift_start(self, which, values):
        """The posture as a lift begins; which = "first" (the episode's first) or "later"."""
        self.lift_start_count[which] += 1
        for k, v in values.items():
            self.lift_starts[which][k] = self.lift_starts[which].get(k, 0.0) + v

    def record_hold(self, side, values):
        """One step of standing on `side` ("L"/"R") with the other foot held up."""
        self.hold_steps[side] += 1
        for k, v in values.items():
            self.holds[side][k] = self.holds[side].get(k, 0.0) + v

    def record_support(self, left_down, right_down):
        side = "L" if left_down and not right_down else "R" if right_down and not left_down else None
        if side:
            self.single_steps[side] += 1
            if side != self._single_side:
                self.single_runs[side] += 1
        self._single_side = side

    def record_touchdown(self, step_length, impact_speed):
        self.touchdowns += 1
        self.step_lengths.append(step_length)
        self.impacts.append(impact_speed)

    def record(self, stage, parts, metrics):
        self.steps += 1
        self.stage_steps[stage] = self.stage_steps.get(stage, 0) + 1
        for k, v in parts.items():
            self.parts[k] = self.parts.get(k, 0.0) + v
        for k, v in metrics.items():
            self.metrics[k] = self.metrics.get(k, 0.0) + v

    def count(self, event):
        self.events[event] = self.events.get(event, 0) + 1

    def merge(self, other):
        """Add another WalkStats (e.g. from a parallel worker) into this one."""
        self.steps += other.steps
        for mine, theirs in ((self.parts, other.parts), (self.metrics, other.metrics), (self.events, other.events),
                             (self.stage_steps, other.stage_steps), (self.single_steps, other.single_steps),
                             (self.single_runs, other.single_runs)):
            for k, v in theirs.items():
                mine[k] = mine.get(k, 0) + v
        self.touchdowns += other.touchdowns
        self.step_lengths += other.step_lengths
        self.impacts += other.impacts
        for side in ("L", "R"):
            self.hold_steps[side] += other.hold_steps[side]
            for k, v in other.holds[side].items():
                self.holds[side][k] = self.holds[side].get(k, 0.0) + v
        for which in ("first", "later"):
            self.lift_start_count[which] += other.lift_start_count[which]
            for k, v in other.lift_starts[which].items():
                self.lift_starts[which][k] = self.lift_starts[which].get(k, 0.0) + v

    # (label, key, format) of the per-leg table; sideways values point toward the lifted foot
    HOLD_ROWS = [
        ("on the foot alone (other >= 2 cm up)", "alone", "{:.0%}"),
        ("lifted foot height, cm", "lift_cm", "{:.1f}"),
        ("center of mass from the foot's center, cm (+ = to the lifted)",
         "com_toward_lifted_cm", "{:+.1f}"),
        ("capture point inside the foot", "icp_inside", "{:.0%}"),
        ("capture point margin, cm", "icp_margin_cm", "{:+.1f}"),
        ("pelvis tilt toward the lifted foot, deg", "tilt_toward_lifted_deg", "{:+.1f}"),
        ("pelvis pitch, deg (+ = forward)", "pitch_deg", "{:+.1f}"),
        ("standing leg torque: hip roll", "stance_hip_roll_torque", "{:.0%}"),
        ("standing leg torque: ankle roll", "stance_ankle_roll_torque", "{:.0%}"),
        ("standing leg torque: ankle pitch", "stance_ankle_pitch_torque", "{:.0%}"),
        ("standing leg off target: hip roll, deg", "stance_hip_roll_off_deg", "{:.1f}"),
        ("standing leg off target: ankle roll, deg", "stance_ankle_roll_off_deg", "{:.1f}"),
        ("lifted leg off target: knee, deg", "lifted_knee_off_deg", "{:.1f}"),
        ("lifted leg off target: hip flex, deg", "lifted_hip_flex_off_deg", "{:.1f}"),
    ]

    # (label, key, format) of the first | later lift table, measured from the foot about to be stood on
    LIFT_START_ROWS = [
        ("feet apart, cm", "feet_apart_cm", "{:.1f}"),
        ("other foot ahead of the standing one, cm", "other_foot_ahead_cm", "{:+.1f}"),
        ("feet turned apart, deg (+ = other foot turned left)", "feet_turned_apart_deg", "{:+.1f}"),
        ("CoM from between the feet toward the other foot, cm", "com_toward_other_cm", "{:+.1f}"),
        ("CoM from between the feet forward, cm", "com_ahead_cm", "{:+.1f}"),
        ("pelvis turned since the start, deg", "pelvis_turned_deg", "{:+.1f}"),
        ("standing foot moved since the start, cm", "stance_foot_moved_cm", "{:.1f}"),
        ("other foot moved since the start, cm", "other_foot_moved_cm", "{:.1f}"),
    ]

    def _lift_start_lines(self):
        """The one-leg stage, the episode's first lift | the later ones: falls, and the posture each starts from."""
        n = self.lift_start_count
        if not (n["first"] or n["later"]):
            return []
        def row(label, first, later):
            return f"      {label:<62}{first:>7} | {later}"
        def cell(which, key, fmt):
            return fmt.format(self.lift_starts[which][key] / n[which]) if n[which] else "-"
        def fell(which):
            falls = self.events.get("lift_fall_" + which, 0)
            return f"{falls} ({falls / n[which]:.0%})" if n[which] else "-"
        lines = [row("ONE LEG, the episode's first lift | the later ones:", "first", "later"),
                 row("lifts started", n["first"], n["later"]),
                 row("fell during it (from the weight shift on)", fell("first"), fell("later")),
                 "      as the weight shift begins:"]
        return lines + [row(label, cell("first", key, fmt), cell("later", key, fmt))
                        for label, key, fmt in self.LIFT_START_ROWS]

    def _hold_lines(self, control_dt):
        """The one-leg stage per leg: standing on the left foot | standing on the right foot."""
        if not (self.hold_steps["L"] or self.hold_steps["R"]):
            return []

        def cell(side, key, fmt):
            n = self.hold_steps[side]
            return fmt.format(self.holds[side][key] / n) if n else "-"
        ev = self.events
        def row(label, left, right):
            return f"      {label:<62}{left:>7} | {right}"
        lines = [row("ONE LEG, per standing foot:", "left", "right"),
                 row("hold time recorded, s", f"{self.hold_steps['L'] * control_dt:.1f}",
                     f"{self.hold_steps['R'] * control_dt:.1f}"),
                 row("falls while holding the other foot up", ev.get("where_hold_L", 0), ev.get("where_hold_R", 0)),
                 row("falls while lifting / lowering the other foot", ev.get("where_lift_L", 0), ev.get("where_lift_R", 0))]
        lines += [row(label, cell("L", key, fmt), cell("R", key, fmt)) for label, key, fmt in self.HOLD_ROWS]
        other = {k[6:]: v for k, v in ev.items() if k in ("where_still", "where_shift")}
        if other:
            lines.append("      other falls: " + ", ".join(f"{k} {v}" for k, v in other.items()))
        return lines

    @staticmethod
    def _motor_lines(m, joint_order):
        """Every motor of both legs, left|right: torque use, distance from its commanded angle, movement."""
        if f"motor_torque_L_{joint_order[0]}" not in m:
            return []
        def pair(kind, j, fmt):
            return f"{j} {fmt(m[f'motor_{kind}_L_{j}'])}|{fmt(m[f'motor_{kind}_R_{j}'])}"
        lines = [
            "    motors left|right: torque " + " ".join(pair("torque", j, lambda v: f"{v:.0%}") for j in joint_order),
            "      off target (deg) " + " ".join(pair("error", j, lambda v: f"{v:.1f}") for j in joint_order)
            + " | moves (deg/s) " + " ".join(pair("moves", j, lambda v: f"{v:.0f}") for j in joint_order),
        ]
        # A motor that doesn't follow its command, or stays still while its twin on the other leg works.
        problems = []
        for j in joint_order:
            for side, other in (("L", "R"), ("R", "L")):
                err, moves = m[f"motor_error_{side}_{j}"], m[f"motor_moves_{side}_{j}"]
                if err > 10.0:
                    problems.append(f"{side} {j} is {err:.0f} deg off its target on average")
                if moves < 0.2 * m[f"motor_moves_{other}_{j}"] and m[f"motor_moves_{other}_{j}"] > 5.0:
                    problems.append(f"{side} {j} barely moves ({moves:.1f} deg/s, other leg {m[f'motor_moves_{other}_{j}']:.0f})")
        if problems:
            lines.append("      MOTOR WARNING: " + "; ".join(problems))
        return lines

    def summary(self, control_dt, joint_order, servo, curriculum_line=""):
        if self.steps == 0:
            return "    no steps"
        m = {k: v / self.steps for k, v in self.metrics.items()}
        p = {k: v / self.steps for k, v in self.parts.items()}
        seconds = self.steps * control_dt
        falls = ", ".join(f"{k[5:]} {v}" for k, v in sorted(self.events.items()) if k.startswith("fall_")) or "none"
        stages = ", ".join(f"{k} {v / self.steps:.0%}" for k, v in sorted(self.stage_steps.items()))

        def stretch(side):
            return self.single_steps[side] / max(1, self.single_runs[side]) * control_dt
        single_total = (self.single_steps["L"] + self.single_steps["R"]) / self.steps
        step_len = sum(self.step_lengths) / len(self.step_lengths) if self.step_lengths else 0.0
        impact = sum(self.impacts) / len(self.impacts) if self.impacts else 0.0
        battery_min = 3.0 * 60.0 / max(m["current"], 1e-3)   # 3000 mAh pack
        lines = [
            f"    time by stage: {stages}" + (f" | {curriculum_line}" if curriculum_line else ""),
            f"    speed {m['v_forward']:+.3f} m/s (asked {m['command']:.3f}), sideways {m['v_side_abs']:.3f}, "
            f"facing {m['heading_error']:.0f} deg off | height {m['height']:.3f} m | pelvis roll {m['roll']:+.1f} "
            f"pitch {m['pitch']:+.1f} deg | feet apart {m['stance_width'] * 100:.1f} cm, one ahead of the other "
            f"{m.get('feet_stagger_abs', 0) * 100:.1f} cm (left ahead by {m.get('feet_stagger', 0) * 100:+.1f} on average)",
            f"    stability: capture point over the feet {m['icp_inside']:.0%} ({m['icp_margin'] * 100:+.1f} cm), "
            f"CoM over the feet {m['com_inside']:.0%} ({m['com_margin'] * 100:+.1f} cm)",
            f"    stepping: on one foot {single_total:.0%}, stretch L {stretch('L'):.2f} s R {stretch('R'):.2f} s "
            f"(human walk ~0.4) | {self.touchdowns / seconds:.1f} steps/s, {step_len * 100:.1f} cm per step | "
            f"foot slip {m['slip'] * 100:.1f} cm/s | landing {impact:.2f} m/s",
            f"    motors: torque (avg of max) " + " ".join(f"{j} {m['torque_' + j]:.0%}" for j in joint_order)
            + f" | peak {m['torque_peak']:.0%} | above rated {m['over_rated']:.0%} of the time | at a joint limit "
            f"{m['at_limit']:.0%} | ~{m['current']:.1f} A, {m['current'] * servo.voltage:.0f} W, "
            f"~{battery_min:.0f} min per 3000 mAh",
            *self._motor_lines(m, joint_order),
            *self._hold_lines(control_dt),
            *self._lift_start_lines(),
            f"    policy: correction to reference |a| {m['residual']:.2f} | falls: {falls}",
            f"    reward terms/step (penalties leave {m.get('alive_kept', 1):.0%} of the alive bonus) = " + " ".join(f"{k} {v:+.3f}" for k, v in p.items()),
        ]
        return "\n".join(lines)
