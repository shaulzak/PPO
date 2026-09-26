"""
The report's per-leg table for the one-leg stage (standing on the left foot | on the right foot): holds and
falls are kept per side, merged across parallel workers, and printed in the right column.
    python tests/test_walk_stats.py
"""
from _common import check, finish

from walk_metrics import WalkStats

print("test_walk_stats")

keys = [key for _, key, _ in WalkStats.HOLD_ROWS]
a, b = WalkStats(), WalkStats()
for _ in range(10):
    a.record_hold("L", {k: 1.0 for k in keys})
for _ in range(30):
    b.record_hold("R", {k: 0.5 for k in keys})
a.count("where_hold_L")
b.count("where_hold_R")
b.count("where_hold_R")
a.merge(b)
check("hold steps are kept per standing foot and merged", a.hold_steps == {"L": 10, "R": 30})
check("measures are averaged per side", a.holds["L"]["lift_cm"] / a.hold_steps["L"] == 1.0
      and a.holds["R"]["lift_cm"] / a.hold_steps["R"] == 0.5)
lines = a._hold_lines(control_dt=0.02)
text = "\n".join(lines)
print(text)
check("the table has a left | right header", "left | right" in lines[0])
falls = next(l for l in lines if "falls while holding" in l)
check("falls land in their side's column", falls.rstrip().endswith("1 | 2"), falls)
hold = next(l for l in lines if "hold time" in l)
check("hold time per side (10 and 30 steps of 20 ms)", hold.rstrip().endswith("0.2 | 0.6"), hold)
check("the columns line up", len({l.index("|") for l in lines}) == 1, text)
check("a side with no holds prints '-'", "-" in "\n".join(WalkStats()._hold_lines(0.02)) or
      not WalkStats()._hold_lines(0.02))
empty_left = WalkStats()
empty_left.record_hold("R", {k: 1.0 for k in keys})
check("a side with no holds shows '-' in its column", "- | " in "\n".join(empty_left._hold_lines(0.02)))

finish()
