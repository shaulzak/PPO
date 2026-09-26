"""
The stage plan: only test episodes count, a stage passes after enough good ones and enough time (and only
once more time stops helping), the state is saved.
    python tests/test_curriculum.py
"""
import os
import tempfile

from _common import check, finish

import curriculum as C

print("test_curriculum")

state = os.path.join(tempfile.gettempdir(), "biped_test_curriculum.json")
if os.path.exists(state):
    os.remove(state)
cur = C.Curriculum(state)
progress = cur.progress_path
if os.path.exists(progress):
    os.remove(progress)
for name in ("stand", "shift", "one_leg"):     # the time rules are tested separately below
    cur.add_time(name, C.MAX_STAGE_MINUTES * 60)
good = {"survived": 1.0, "shift_track": 1.0, "on_left_foot": 1.0, "on_right_foot": 1.0, "single_stretch": 1.0, "single_fraction": 1.0,
        "speed_ratio": 1.0, "still_stable": 1.0,
        "survived_on_left": 1.0, "survived_on_right": 1.0}

for _ in range(100):
    cur.episode_done("stand", good, is_test=False)
check("training episodes never pass a stage", cur.level == 0)

for _ in range(C.MIN_EPISODES - 1):
    cur.episode_done("stand", good, is_test=True)
check("not passed before MIN_EPISODES test episodes", cur.level == 0)
cur.episode_done("stand", good, is_test=True)
check("passed after MIN_EPISODES good test episodes", cur.level == 1 and cur.stage.name == "shift")

bad = dict(good, survived=0.5)
for _ in range(C.WINDOW):
    cur.episode_done("shift", bad, is_test=True)
check("a stage with too many falls isn't passed", cur.level == 1)

for _ in range(C.WINDOW):   # push the bad ones out of the window
    cur.episode_done("shift", good, is_test=True)
check("shift passed", cur.level == 2 and cur.stage.name == "one_leg")
weak_leg = dict(good, on_left_foot=0.4)
for _ in range(C.WINDOW):
    cur.episode_done("one_leg", weak_leg, is_test=True)
check("one leg isn't passed while one of the legs is weak", cur.level == 2)

for _ in range(C.WINDOW):
    cur.episode_done("one_leg", dict(good, still_stable=0.3), is_test=True)
check("one leg isn't passed while it can't stand still between the holds", cur.level == 2)
for _ in range(C.WINDOW):
    cur.episode_done("stand", bad, is_test=True)          # forgot how to stand
for _ in range(C.WINDOW):
    cur.episode_done("one_leg", good, is_test=True)
check("a stage isn't passed while an earlier stage is lost", cur.level == 2)
check("the report shows the lost earlier stage", "stand LOST" in cur.report_line(), cur.report_line())
for _ in range(C.WINDOW):
    cur.episode_done("stand", good, is_test=True)
cur.episode_done("one_leg", good, is_test=True)
check("one leg isn't passed before the pauses wait for the full steady streak",
      cur.level == 2 and cur.steady_hold < C.STEADY_HOLD_GOAL, f"streak {cur.steady_hold:.2f} s")
check("the report shows the steady streak", "pauses wait for 0.20 s" in cur.report_line(), cur.report_line())

# the steady streak: starts at one control step, grows while most pauses reach it in time, shrinks if not
check("the steady streak starts at 0.2 s (not one good moment)", abs(cur.steady_hold - 0.2) < 1e-9)
for _ in range(C.STEADY_WINDOW - 1):
    cur.steady_pause_done(True)
check("the streak isn't changed before a full window of pauses", abs(cur.steady_hold - 0.2) < 1e-9)
cur.steady_pause_done(True)
check("the streak grows ~x1.25 in whole control steps (0.2 -> 0.24)", abs(cur.steady_hold - 0.24) < 1e-9, f"{cur.steady_hold:.3f}")
for _ in range(C.STEADY_WINDOW):
    cur.steady_pause_done(False)
check("the streak shrinks back when most pauses miss it", abs(cur.steady_hold - 0.2) < 1e-9, f"{cur.steady_hold:.3f}")
for _ in range(C.STEADY_WINDOW):
    cur.steady_pause_done(False)
check("never below the start", abs(cur.steady_hold - 0.2) < 1e-9)
for _ in range(60 * C.STEADY_WINDOW):
    cur.steady_pause_done(True)
check("it grows up to the goal and stops there", abs(cur.steady_hold - C.STEADY_HOLD_GOAL) < 1e-9,
      f"{cur.steady_hold:.3f}")
check("the streak is saved and reloaded", abs(C.Curriculum(state).steady_hold - C.STEADY_HOLD_GOAL) < 1e-9)
check("without a training state (Run.py, the robot) the pauses wait for the goal",
      C.Curriculum().steady_hold == C.STEADY_HOLD_GOAL)
cur.episode_done("one_leg", good, is_test=True)
check("passed once the earlier stage is back", cur.level == 3)

check("the reached stage is saved and reloaded", C.Curriculum(state).level == 3)
tests = sum(cur.choose()[1] for _ in range(4000)) / 4000
check("about TEST_SHARE of the episodes are tests", abs(tests - C.TEST_SHARE) < 0.03, f"{tests:.3f}")
check("the test windows are saved and reloaded (earlier stages are judged right after a resume)",
      C.Curriculum(state).summary("stand")["episodes"] == C.WINDOW)
check("stage minutes are saved and reloaded", C.Curriculum(state).minutes["shift"] >= C.MAX_STAGE_MINUTES)
check("every test episode is written to stage_progress.csv",
      sum(1 for _ in open(progress, encoding="utf-8")) > 100)
os.remove(state)
os.remove(progress)

# ----- time per stage -----
def run(still_at, minutes, always_improving=False):
    """A stand stage tested once per simulated minute; still_stable given by still_at(minute)."""
    c = C.Curriculum()
    if always_improving:
        c.settled = lambda name: False
    passed_at = None
    for minute in range(1, minutes + 1):
        c.add_time("stand", 60)
        c.episode_done("stand", dict(good, still_stable=still_at(minute)), is_test=True)
        if c.level == 1 and passed_at is None:
            passed_at = minute
    return c, passed_at

c, at = run(lambda m: 1.0, C.MIN_STAGE_MINUTES - 1)
check("not passed before MIN_STAGE_MINUTES, even with perfect tests", at is None)
check("the time line says more time isn't helping", "flat" in c.time_line(), c.time_line())
c, at = run(lambda m: 1.0, C.MIN_STAGE_MINUTES + 5)
check("passed right after MIN_STAGE_MINUTES when the score is flat", at == C.MIN_STAGE_MINUTES, f"at {at}")
improving = lambda m: min(1.0, 0.7 + 0.004 * m)   # passing, but still getting better until minute 75+
c, at = run(improving, C.MIN_STAGE_MINUTES + 5)
check("kept while the score still improves", at is None, c.time_line())
check("the time line says more time helps", "improving" in c.time_line(), c.time_line())
c, at = run(improving, C.MAX_STAGE_MINUTES)
check("passed once the improvement stops", at is not None and at < C.MAX_STAGE_MINUTES, f"at {at}")
worse = lambda m: 1.0 if m < 75 else max(0.72, 1.0 - 0.02 * (m - 75))   # like the first stand run
c, at = run(worse, C.MIN_STAGE_MINUTES + 5)
check("not passed while the score is getting worse, even above the targets", at is None, c.time_line())
check("the time line says it's getting worse", "getting worse" in c.time_line(), c.time_line())
c, at = run(worse, C.MAX_STAGE_MINUTES)
check("passed once the score settles again", at is not None and at < C.MAX_STAGE_MINUTES, f"at {at}")
c, at = run(lambda m: 1.0, C.MAX_STAGE_MINUTES, always_improving=True)
check("passed at MAX_STAGE_MINUTES even if it keeps improving", at == C.MAX_STAGE_MINUTES, f"at {at}")

# ----- per leg -----
legs = C.Curriculum()
legs.level = 2                                             # one_leg
never_left = dict(good, survived_on_left=None)             # no test got to standing on the left foot
for _ in range(C.WINDOW):
    legs.episode_done("one_leg", never_left, is_test=True)
check("a foot never tested doesn't pass", not legs.leg_passed(0) and legs.leg_passed(1))
check("unmeasured values are left out of the average, not counted as 0",
      legs.summary("one_leg")["survived_on_left"] is None and legs.summary("one_leg")["survived"] == 1.0)
check("the stage doesn't pass with one foot untested", legs.level == 2)
check("training then practises only the untested / weaker foot", legs.weaker_leg() == 0)
check("the report says which foot is practised", "only standing on the left foot" in legs.report_line())
for _ in range(C.WINDOW):
    legs.episode_done("one_leg", dict(good, on_right_foot=0.3), is_test=True)
check("and switches when the other foot is the weak one", legs.weaker_leg() == 1)
for _ in range(C.WINDOW):
    legs.episode_done("one_leg", dict(good, on_left_foot=0.3, on_right_foot=0.3), is_test=True)
check("both feet weak: practise both", legs.weaker_leg() is None)
check("a stage score leaves unmeasured values out (not as a zero)",
      C.STAGES[2].score(never_left) > C.STAGES[2].score(dict(good, survived_on_left=0.0)))

swaying = C.Curriculum()
for _ in range(C.WINDOW):
    swaying.episode_done("stand", dict(good, still_stable=0.3), is_test=True)
check("standing isn't passed by not falling while swaying", swaying.level == 0)

finish()
