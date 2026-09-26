"""
How one step's reward terms are combined: never below zero (living on beats falling), and every penalty
always changes the reward (no floor where "a bit worse" and "much worse" both give 0).
    python tests/test_reward.py
"""
from _common import check, finish

from types import SimpleNamespace

import motions as M
import WalkAgent as W
from curriculum import Curriculum
from WalkAgent import combine_reward, facing_term, stillness_term, quiet_motors_term

print("test_reward")

base = {"alive": 1.5, "imitate": 0.4, "task": 0.3}
reward, kept = combine_reward(base)
check("no penalties: positive terms + the full alive bonus", abs(reward - 2.2) < 1e-9 and kept == 1.0, f"{reward:.3f}")

heavy = dict(base, drift=-0.5, facing=-0.5, stagger=-0.5, slip=-0.3, smooth=-0.2)
worse = dict(heavy, upright=-0.3)
r_heavy, _ = combine_reward(heavy)
r_worse, _ = combine_reward(worse)
check("never below zero, even with every penalty at its maximum", combine_reward(dict(heavy, height=-50))[0] >= 0)
check("heavy penalties still rank: one more penalty lowers the reward", r_worse < r_heavy,
      f"{r_heavy:.3f} -> {r_worse:.3f}")
check("more task reward always helps, penalties or not", combine_reward(dict(heavy, task=0.8))[0] > r_heavy)
check("a negative task value counts as a penalty", combine_reward(dict(base, task=-0.5))[0] < combine_reward(base)[0])

# facing: a bonus that pulls back at every angle, not a penalty that eats the alive bonus
turned = [facing_term(d) for d in (0, 10, 20, 34, 45, 60, 90, 180)]
check("facing: every degree more turned lowers the reward, out to 90 deg", all(a > b for a, b in zip(turned, turned[1:-1])),
      ", ".join(f"{t:+.2f}" for t in turned))
check("facing: turning up to 45 deg leaves the whole alive bonus", combine_reward(dict(base, facing=facing_term(44)))[1] == 1.0)
check("facing: turning left or right counts the same", facing_term(-30) == facing_term(30))

# standing still: flat feet and a still pelvis, not rocking from foot edge to foot edge
check("stillness: flat and not moving is the most", stillness_term(1.0, 0.0) == 1.0)
check("stillness: rocking on the foot edges (6 of 8 corners down, 0.1 m/s) pays nothing",
      stillness_term(6 / 8, 0.1) == 0.0)
check("stillness: flatter feet and a slower pelvis always pay more",
      stillness_term(1.0, 0.02) > stillness_term(0.75, 0.02) and stillness_term(1.0, 0.02) > stillness_term(1.0, 0.05))

check("stillness: the ~5 Hz tremor (5.6 cm/s, tilting 0.35 rad/s) pays well under half",
      stillness_term(1.0, 0.056, 0.35) < 0.5 * stillness_term(1.0, 0.0, 0.0))
check("stillness: tilting less always pays more", stillness_term(1.0, 0.02, 0.1) > stillness_term(1.0, 0.02, 0.3))

# quiet motors: a straight line in the average motor speed, +1 not moving, 0 at 20 deg/s, -1 from 40 deg/s
quiet = [quiet_motors_term([v] * 12) for v in range(0, 41)]
check("quiet: motors not moving is the most", quiet[0] == 1.0)
check("quiet: continuous - every deg/s less pays exactly 0.05 more, no jumps, out to 40 deg/s",
      all(abs((a - b) - 0.05) < 1e-9 for a, b in zip(quiet, quiet[1:])))
check("quiet: the ~5 Hz tremor (~18 deg/s measured by probe_still) pays only ~0.1",
      abs(quiet_motors_term([18.0] * 12) - 0.1) < 1e-9)
check("quiet: the average over the motors counts (one motor trembling alone costs less than all)",
      quiet_motors_term([18.0] + [0.0] * 11) > quiet_motors_term([18.0] * 12))
check("quiet: floored at -1, moving either way counts the same",
      quiet_motors_term([500.0] * 12) == -1.0 and quiet_motors_term([-10.0] * 12) == quiet_motors_term([10.0] * 12))
check("quiet: even at its worst it leaves over a third of the alive bonus",
      combine_reward(dict(base, quiet=quiet_motors_term([500.0] * 12)))[1] > 1 / 3)

# the one-leg pauses wait for a streak of standing well (the user's rule: points bank only when the goal is
# reached, a broken streak goes back to what was banked, waiting itself pays nothing)
check("standing well: flat, still, not turning", W.standing_well([1.0] * 8, 0.05, 0.01, 0.05, 0.02))
check("standing well: not with one corner up, the tremor (5.6 cm/s) or turning",
      not W.standing_well([1.0] * 7 + [0.0], 0.05, 0.01, 0.05, 0.02)
      and not W.standing_well([1.0] * 8, 0.05, 0.056, 0.05, 0.02)
      and not W.standing_well([1.0] * 8, 0.05, 0.01, 0.05, 0.3))


def pause_agent(t, hold):
    cur = Curriculum()
    cur.steady_hold = hold
    return SimpleNamespace(stage=W.STAGES[2], steady_pauses=True, motion_time=t, control_dt=W.CONTROL_DT,
                           curriculum=cur, is_test=False, waited=0.0, steady_steps=0, steady_points=0.0,
                           standing_well_now=False)


def steps(agent, wells):
    rewards = []
    for well in wells:
        assert W.WalkAgent._waiting(agent)
        agent.standing_well_now = well
        rewards.append(W.WalkAgent._steady_step(agent))
    return rewards


a = pause_agent(W.ONE_LEG_SETTLE + 0.1, hold=0.1)                   # 5 control steps
check("the pause waits from ONE_LEG_SETTLE s in, not before",
      W.WalkAgent._waiting(a) and not W.WalkAgent._waiting(pause_agent(W.ONE_LEG_SETTLE - 0.05, 0.1)))
r = steps(a, [False] * 200)
check("waiting while not standing well pays nothing and the motion doesn't go on (no time limit)",
      sum(r) == 0.0 and a.motion_time == W.ONE_LEG_SETTLE + 0.1)
r = steps(a, [True] * 4 + [False])
check("a broken streak goes back to what was banked (the sum is 0 again)",
      r[:4] == [W.STEADY_POINT] * 4 and abs(sum(r)) < 1e-9, f"{r}")
r = steps(a, [True] * 5)
check("a full streak banks its points and the motion goes on to the weight shift",
      abs(sum(r) - 5 * W.STEADY_POINT) < 1e-9 and abs(a.motion_time - M.ONE_LEG_PAUSE) < 1e-9
      and not W.WalkAgent._waiting(a), f"{sum(r):.1f}, motion at {a.motion_time:.2f} s")
check("that pause counts as reached late (it waited over 2 s)", list(a.curriculum.steady_pauses) == [False])
b = pause_agent(M.ONE_LEG_CYCLE / 2 + W.ONE_LEG_SETTLE, hold=0.02)
steps(b, [False] * 3 + [True])
check("the second half's pause goes on to its own shift, in time",
      abs(b.motion_time - (M.ONE_LEG_CYCLE / 2 + M.ONE_LEG_PAUSE)) < 1e-9 and list(b.curriculum.steady_pauses) == [True])
c = pause_agent(0.6, 0.1)
c.stage = W.STAGES[0]
check("other stages never wait", not W.WalkAgent._waiting(c))

# exploration noise: low only while it should stand still (STILL_NOISE_STD), the learned one elsewhere
trainer = SimpleNamespace(noise_std=lambda: 0.15)
n = SimpleNamespace(stage=W.STAGES[2], motion_time=0.2, episode_time=5.0, last_push=-1e9, trainer=trainer)
at = lambda t: (setattr(n, "motion_time", t), W.WalkAgent._noise_scale(n))[1]
check("noise: a one-leg pause gets STILL_NOISE_STD", abs(at(0.2) * 0.15 - W.STILL_NOISE_STD) < 1e-9
      and abs(at(M.ONE_LEG_CYCLE / 2 + 2.0) * 0.15 - W.STILL_NOISE_STD) < 1e-9)
check("noise: shifting, lifting, holding keep the full noise",
      all(at(M.ONE_LEG_PAUSE + x) == 1.0 for x in (0.5, M.ONE_LEG_LEAN + 1.0, M.ONE_LEG_LEAN + M.ONE_LEG_LIFT + 1.0)))
n.stage, n.episode_time, n.last_push = W.STAGES[0], 5.0, 3.0
check("noise: the stand stage gets it from 1 s after a push", abs(W.WalkAgent._noise_scale(n) * 0.15 - W.STILL_NOISE_STD) < 1e-9)
n.last_push = 4.5
check("noise: full while recovering from a push", W.WalkAgent._noise_scale(n) == 1.0)
n.stage = W.STAGES[1]
check("noise: other stages keep the full noise", W.WalkAgent._noise_scale(n) == 1.0)

finish()
