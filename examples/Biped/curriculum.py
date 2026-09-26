"""
Training stages, from standing still to walking. One network learns them all (the current stage is part
of its observation); the robot moves on when the current stage's pass test holds over recent episodes,
it has spent enough time on the stage, and more time no longer helps. It keeps rehearsing (and re-testing)
earlier stages so it doesn't forget them.

    stand    stand still on both feet, recovering from pushes
    shift    move the weight from foot to foot without lifting them
    one_leg  shift onto one foot, lift the other and hold it up, then the other side
    march    step in place, one foot at a time
    walk     walk forward at the commanded speed
"""
import json
import os
import random
from collections import deque
from dataclasses import dataclass, field

from motions import TASKS


@dataclass
class Stage:
    name: str
    seconds: float                      # episode length
    push_speed: float                   # m/s: largest sideways/forward kick given to the pelvis now and then
    command_range: tuple                # m/s forward (walk only)
    weights: dict                       # reward term -> weight (terms missing here are off)
    requirement: str                    # the pass test, in words (for the report)
    targets: dict = field(default_factory=dict)   # test result -> value it must reach to pass
    final: bool = False                 # the last stage is never passed

    def passed(self, r):
        # a value that was never measured (None: e.g. no test reached that foot yet) doesn't pass
        return not self.final and all(r.get(k) is not None and r[k] >= v for k, v in self.targets.items())

    def score(self, r):
        """How close a test is to the pass targets: 1.0 = exactly at them (each term capped at 1.5, so a
        passed stage still shows whether it keeps getting better). Unmeasured values are left out."""
        terms = [min(1.5, r[k] / v) for k, v in self.targets.items() if r.get(k) is not None]
        return sum(terms) / len(terms) if terms else 0.0


# upright 3: at 1 the robot drifted into a 14 deg forward lean while standing.
# slip 1: at 2 it was the biggest term (mostly exploration jitter) and pushed the reward per step below zero.
# residual: small cost on the size of the correction to the reference motion.
COMMON = {"upright": 3.0, "slip": 1.0, "impact": 0.5, "energy": 0.05, "limit": 0.2, "smooth": 0.5, "yaw": 0.05,
          "residual": 0.3}

# Episode lengths and motions are slow on purpose (motions.py): time to settle at every step.
STAGES = [
    Stage("stand", 15.0, 0.4, (0.0, 0.0),
          dict(COMMON, alive=0.5, balance=0.5, imitate=0.5, still=1.0, task=1.0, height=5.0, stagger=0.5,
               facing=0.5),
          # task: WalkAgent.stillness_term (flat feet, pelvis not moving) between the pushes.
          # Not falling isn't enough: between the pushes it has to actually stand still (it used to pass
          # while swaying, and carried the sway into the later stages).
          "no falls (>= 99% of the tests) in 15 s episodes with pushes, and stands still and stable >= 70% of the time "
          "between pushes",
          dict(survived=0.99, still_stable=0.7)),
    Stage("shift", 15.0, 0.15, (0.0, 0.0),
          dict(COMMON, alive=0.5, balance=0.5, imitate=0.5, task=1.0, drift=0.5, height=5.0, stagger=0.5, facing=0.5),
          "no falls (>= 99%) and follows the weight shift (score >= 0.6)",
          dict(survived=0.99, shift_track=0.6)),
    # Two full cycles (motions.ONE_LEG_CYCLE 22.4 s): each leg is held up twice, every hold starts from
    # standing still.
    Stage("one_leg", 45.0, 0.1, (0.0, 0.0),
          # alive 1.5: with 0.5 most steps summed below zero and were cut by the zero floor, so "almost
          # lifted" and "never tried" both scored 0 and the policy got no gradient toward lifting.
          dict(COMMON, alive=1.5, balance=0.7, imitate=0.5, task=2.5, drift=0.5, height=3.0, slip=0.5, stagger=0.5, facing=0.5),
          # No "quiet" (WalkAgent.quiet_motors_term): tried at 1.0 in the pauses, the tremor got worse.
          "no falls (>= 99%, and on EACH foot), is on the standing foot alone (the other >= 2 cm up) >= 70% of the hold time on "
          "EACH leg, and stands "
          "stable on both feet >= 70% of the pauses between",
          dict(survived=0.99, on_left_foot=0.7, on_right_foot=0.7, still_stable=0.7,
               survived_on_left=0.99, survived_on_right=0.99)),
    Stage("march", 16.0, 0.1, (0.0, 0.0),
          dict(COMMON, alive=0.4, balance=0.5, imitate=0.7, gait=0.8, drift=0.5, height=2.0, heading=0.3),
          "no falls (>= 99%), one-foot stretches >= 0.3 s and >= 40% of the time on one foot",
          dict(survived=0.99, single_stretch=0.3, single_fraction=0.4)),
    Stage("walk", 30.0, 0.15, (0.03, 0.12),
          dict(COMMON, alive=0.3, balance=0.4, imitate=0.6, gait=0.6, speed=1.5, side=0.75, height=2.0,
               heading=0.3),
          "final stage (score: no falls (>= 99%), speed >= 80% of the command)",
          dict(survived=0.99, speed_ratio=0.8), final=True),
]
assert [s.name for s in STAGES] == TASKS

# The pass test uses test episodes: best action, no exploration noise, no learning (what Run.py and the
# real robot will do). Judging by training episodes kept a stage that was already mastered failing,
# because the exploration noise alone knocked the robot over.
TEST_SHARE = 0.15      # share of episodes run as tests of the current stage
# The user's rule: no falls. 99% of a 20-test window means not a single fall in the last 20 tests.
WINDOW = 20            # test episodes per stage used for the pass test
MIN_EPISODES = 15      # before a stage can be passed
REHEARSAL = 0.2        # share of training episodes spent on an earlier (already passed) stage
# Share of the test episodes spent re-testing an earlier stage: a new stage is passed only if the robot can
# still do the earlier ones (e.g. still stand stably on both feet after learning to stand on one).
REHEARSAL_TESTS = 0.25
EARLIER_MIN_TESTS = 5  # an earlier stage is judged only once it has this many tests

# Time per stage, in simulated minutes of that stage's own episodes. Stages used to pass within minutes, and
# the habits learned in a hurry (swaying, one favorite leg) carried over. Now a stage is kept at least
# MIN_STAGE_MINUTES, and after reaching its targets it stays while the test score is still clearly
# improving (more time still helps) or getting worse (the first stand run passed at minute 92 while standing
# still had dropped from 1.00 to ~0.6 - just above the target on average), up to MAX_STAGE_MINUTES.
MIN_STAGE_MINUTES = 90
MAX_STAGE_MINUTES = 360
TREND_MINUTES = 30     # the trend compares the test score of the last 30 min with the 30 min before
TREND_STEP = 0.03      # score change that counts as "better" / "worse"
TREND_MIN_TESTS = 5    # tests needed in each of the two spans


# One-leg pauses wait until the robot has stood well this long in a row (WalkAgent.standing_well). The
# user's goal is 1 s; the streak starts at 0.2 s and grows once most pauses reach it in time (within the old
# 2 s of waiting), and shrinks back (never below the start) if most don't. The one-leg stage can't pass
# before the streak is at the goal. It started at one control step (0.02 s) - the user: then it learns to
# find one good moment and move on, not to keep standing well.
STEADY_HOLD_START = 0.2
STEADY_HOLD_STEP = 0.02    # one control step: the streak is always whole steps
STEADY_HOLD_GOAL = 1.0
STEADY_WINDOW = 40         # training pauses judged together
STEADY_GROW = 0.75         # share reached in time -> the streak x STEADY_FACTOR
STEADY_SHRINK = 0.25       # share reached in time -> the streak / STEADY_FACTOR
STEADY_FACTOR = 1.25


class Curriculum:
    STAGES_NAMES = [s.name for s in STAGES]

    def __init__(self, state_path=None):
        self.state_path = state_path
        # Without a training state (Run.py, Evaluate.py, probes): the goal itself, like the real robot.
        self.steady_hold = STEADY_HOLD_START if state_path else STEADY_HOLD_GOAL
        self.steady_pauses = deque(maxlen=STEADY_WINDOW)
        self.level = 0
        self.results = {s.name: deque(maxlen=WINDOW) for s in STAGES}
        self.log = []
        self.minutes = {s.name: 0.0 for s in STAGES}   # simulated minutes of each stage's episodes
        self.history = {s.name: [] for s in STAGES}    # (that stage's minutes at the test, test score)
        self.progress_path = None
        if state_path:
            self.progress_path = os.path.join(os.path.dirname(os.path.abspath(state_path)), "stage_progress.csv")
            if os.path.exists(state_path):
                with open(state_path, encoding="utf-8") as f:
                    state = json.load(f)
                self.level = min(int(state.get("level", 0)), len(STAGES) - 1)
                self.minutes.update(state.get("minutes", {}))
                self.steady_hold = max(STEADY_HOLD_START, float(state.get("steady_hold", self.steady_hold)))
                for name, h in state.get("history", {}).items():
                    self.history[name] = [tuple(x) for x in h]
                for name, episodes in state.get("results", {}).items():
                    self.results[name].extend(episodes)

    @property
    def stage(self):
        return STAGES[self.level]

    def choose(self):
        """(stage, is_test) for the next episode."""
        if random.random() < TEST_SHARE:
            if self.level > 0 and random.random() < REHEARSAL_TESTS:
                return STAGES[random.randrange(self.level)], True
            return self.stage, True
        if self.level > 0 and random.random() < REHEARSAL:
            return STAGES[random.randrange(self.level)], False
        return self.stage, False

    def summary(self, name):
        episodes = list(self.results[name])
        if not episodes:
            return None
        summary = {}
        for k in dict.fromkeys(k for e in episodes for k in e):     # tests saved before a key existed lack it
            values = [e[k] for e in episodes if e.get(k) is not None]   # None: not measured in that test
            summary[k] = sum(values) / len(values) if values else None
        return summary | {"episodes": len(episodes)}

    def leg_passed(self, side):
        """One-leg stage: standing on this foot (0 left, 1 right) meets its own targets in the recent tests."""
        s = self.summary("one_leg")
        if not s or s["episodes"] < MIN_EPISODES:
            return False
        name = ("left", "right")[side]
        on, survived = s.get(f"on_{name}_foot"), s.get(f"survived_on_{name}")
        return on is not None and on >= 0.7 and survived is not None and survived >= 0.99

    def weaker_leg(self):
        """The standing foot to practise alone: the one that doesn't pass while the other does, else None."""
        left, right = self.leg_passed(0), self.leg_passed(1)
        if left != right:
            return 1 if left else 0
        return None

    def steady_pause_done(self, in_time):
        """A training one-leg pause ended: the streak was reached in time (True) or not. Adjusts steady_hold."""
        self.steady_pauses.append(bool(in_time))
        if len(self.steady_pauses) < STEADY_WINDOW:
            return
        share = sum(self.steady_pauses) / len(self.steady_pauses)
        old = self.steady_hold
        if share >= STEADY_GROW:
            self.steady_hold = min(STEADY_HOLD_GOAL, self.steady_hold * STEADY_FACTOR)
        elif share <= STEADY_SHRINK:
            self.steady_hold = max(STEADY_HOLD_START, self.steady_hold / STEADY_FACTOR)
        # whole control steps (0.02 s), at least one more / less than before
        self.steady_hold = round(self.steady_hold / STEADY_HOLD_STEP) * STEADY_HOLD_STEP
        if share >= STEADY_GROW and old < STEADY_HOLD_GOAL:
            self.steady_hold = min(STEADY_HOLD_GOAL, max(self.steady_hold, old + STEADY_HOLD_STEP))
        elif share <= STEADY_SHRINK and old > STEADY_HOLD_START:
            self.steady_hold = max(STEADY_HOLD_START, min(self.steady_hold, old - STEADY_HOLD_STEP))
        if self.steady_hold != old:
            message = (f"*** STEADY STREAK: {share:.0%} of the last {STEADY_WINDOW} one-leg pauses stood well in time "
                       f"-> the pauses now wait for {self.steady_hold:.2f} s in a row (goal {STEADY_HOLD_GOAL:.1f})")
            print(message, flush=True)
            self.log.append(message)
            self.steady_pauses.clear()
            self._save()

    def add_time(self, name, seconds):
        """Count the simulated time of a finished episode (training or test) toward its stage."""
        self.minutes[name] += seconds / 60.0

    def trend(self, name):
        """(score of the last TREND_MINUTES, score of the TREND_MINUTES before), or None with too few tests."""
        now = self.minutes[name]
        recent = [sc for t, sc in self.history[name] if t > now - TREND_MINUTES]
        before = [sc for t, sc in self.history[name] if now - 2 * TREND_MINUTES < t <= now - TREND_MINUTES]
        if len(recent) < TREND_MIN_TESTS or len(before) < TREND_MIN_TESTS:
            return None
        return sum(recent) / len(recent), sum(before) / len(before)

    def settled(self, name):
        """The test score has stopped changing: more time neither helps nor hurts right now."""
        t = self.trend(name)
        return t is not None and abs(t[0] - t[1]) < TREND_STEP

    def episode_done(self, name, result, is_test):
        """Record one finished test episode; advance when the current stage passes. Training episodes don't count."""
        if not is_test:
            return
        self.results[name].append(result)
        score = STAGES[self.STAGES_NAMES.index(name)].score(result)
        self.history[name].append((self.minutes[name], score))
        del self.history[name][:-2000]
        self._write_progress(name, result, score)
        current = self.stage
        if name == current.name and not current.final:
            s = self.summary(name)
            minutes = self.minutes[name]
            if (s["episodes"] >= MIN_EPISODES and current.passed(s) and self.earlier_stages_kept()
                    and (name != "one_leg" or self.steady_hold >= STEADY_HOLD_GOAL - 1e-9)
                    and minutes >= MIN_STAGE_MINUTES
                    and (minutes >= MAX_STAGE_MINUTES or self.settled(name))):
                self.level += 1
                message = (f"*** CURRICULUM: passed '{current.name}' ({current.requirement}) after "
                           f"{minutes:.0f} simulated min -> now training '{self.stage.name}'")
                print(message, flush=True)
                self.log.append(message)
        self._save()

    def earlier_stages_kept(self):
        """Every earlier stage still passes its own test (judged on its latest test episodes)."""
        for stage in STAGES[:self.level]:
            s = self.summary(stage.name)
            if s and s["episodes"] >= EARLIER_MIN_TESTS and not stage.passed(s):
                return False
        return True

    def _save(self):
        if self.state_path:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({"level": self.level, "stage": self.stage.name, "minutes": self.minutes,
                           "steady_hold": self.steady_hold,
                           "history": self.history, "results": {k: list(v) for k, v in self.results.items()}}, f)

    def _write_progress(self, name, result, score):
        """One line per test episode (stage_progress.csv), to check later whether more time still helped."""
        if not self.progress_path:
            return
        keys = sorted(result)
        header = ",".join(["stage", "current_stage", "stage_minutes", "total_minutes", "score"] + keys)
        if os.path.exists(self.progress_path):
            with open(self.progress_path, encoding="utf-8") as f:
                old_header = f.readline().strip()
            if old_header != header:              # the measured values changed: keep the old file, start anew
                base, ext = os.path.splitext(self.progress_path)
                n = 1
                while os.path.exists(f"{base}_{n}{ext}"):
                    n += 1
                os.replace(self.progress_path, f"{base}_{n}{ext}")
        new = not os.path.exists(self.progress_path)
        with open(self.progress_path, "a", encoding="utf-8") as f:
            if new:
                f.write(header + "\n")
            f.write(",".join([name, self.stage.name, f"{self.minutes[name]:.2f}",
                              f"{sum(self.minutes.values()):.2f}", f"{score:.3f}"]
                             + ["" if result[k] is None else f"{result[k]:.3f}" for k in keys]) + "\n")

    def time_line(self):
        """How much time each stage got, and whether more time is still helping the current one."""
        spent = ", ".join(f"{s.name} {self.minutes[s.name]:.0f}" for s in STAGES if self.minutes[s.name] > 0)
        name = self.stage.name
        t = self.trend(name)
        if t is None:
            verdict = (f"trend: not enough tests yet (needs {TREND_MIN_TESTS} in each of two "
                       f"{TREND_MINUTES}-min spans)")
        else:
            delta = t[0] - t[1]
            word = ("improving - more time helps" if delta >= TREND_STEP else
                    "getting worse" if delta <= -TREND_STEP else "flat - more time isn't helping now")
            verdict = (f"test score {t[1]:.2f} -> {t[0]:.2f} over the last {2 * TREND_MINUTES} min "
                       f"({delta:+.2f}): {word}")
        return (f"simulated minutes per stage: {spent or 'none yet'} | '{name}': {self.minutes[name]:.0f} of at "
                f"least {MIN_STAGE_MINUTES} (at most {MAX_STAGE_MINUTES}), {verdict} (score 1.00 = at the pass targets)")

    def report_line(self):
        s = self.summary(self.stage.name)
        if not s:
            head = f"stage {self.level + 1}/{len(STAGES)} '{self.stage.name}' (no test episodes yet)"
        else:
            details = ", ".join(f"{k} {'-' if v is None else f'{v:.2f}'}" for k, v in s.items() if k != "episodes")
            earlier = ", ".join(
                f"{st.name} {'?' if e['episodes'] < EARLIER_MIN_TESTS else 'OK' if st.passed(e) else 'LOST'} "
                f"({', '.join(f'{k} {e.get(k) or 0:.2f}' for k in st.targets)}, {e['episodes']} tests)"
                for st in STAGES[:self.level] if (e := self.summary(st.name)))
            head = (f"stage {self.level + 1}/{len(STAGES)} '{self.stage.name}', last {s['episodes']} test episodes: "
                    f"{details} | to pass: {self.stage.requirement}"
                    + (f" | earlier stages: {earlier}" if earlier else ""))
            if self.stage.name == "one_leg":
                reached = (f"{sum(self.steady_pauses) / len(self.steady_pauses):.0%} of the last "
                           f"{len(self.steady_pauses)} training pauses in time" if self.steady_pauses else "no pauses yet")
                head += (f" | pauses wait for {self.steady_hold:.2f} s of standing well in a row (goal "
                         f"{STEADY_HOLD_GOAL:.1f}, needed to pass; {reached})")
                weak = self.weaker_leg()
                head += (" | training episodes practise " +
                         ("both feet" if weak is None else f"only standing on the {('left', 'right')[weak]} foot "
                          f"(the {('right', 'left')[weak]} one passes)"))
        return head + "\n" + self.time_line()
