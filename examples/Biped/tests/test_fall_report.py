"""
A fall by touching the floor (FallSensor -> request_end_episode) is reported to the curriculum as the episode
that fell: its stage, its test flag and its length. The engine's process_end_request() reset the episode
first, so the curriculum got the next, empty episode (0 s long, possibly another stage or test flag).
    python tests/test_fall_report.py
"""
from _common import check, finish, make_world

from bereshit.addons.PPO import Academy
import robot as R
import WalkAgent as W
from curriculum import Curriculum

print("test_fall_report")

Academy.setup_trainer(W.make_config(None))
root, robot = R.build_robot()
floor = R.build_floor()
curriculum = Curriculum()
agent = W.WalkAgent(robot, W.CONTROL_DT, curriculum=curriculum, pushes=False)
root.add_component(agent)
world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)

reports = []
add_time, episode_done = curriculum.add_time, curriculum.episode_done


def record_time(name, seconds):
    reports.append(("time", name, seconds))
    return add_time(name, seconds)


def record_done(name, result, is_test):
    reports.append(("done", name, is_test))
    return episode_done(name, result, is_test)


curriculum.add_time, curriculum.episode_done = record_time, record_done

ticks_per_decision = round(W.CONTROL_DT / W.TICK)
step = 0


def run_until(condition, limit):
    global step
    for _ in range(limit):
        if condition():
            return True
        world.update(step % ticks_per_decision == 0)
        step += 1
    return condition()


# let an episode run 20 decisions without ending on its own
started = run_until(lambda: agent.episode["steps"] >= 20 and not reports, 4000)
if reports:
    reports.clear()
    started = run_until(lambda: agent.episode["steps"] >= 20, 4000) and not reports
check("an episode runs 20 decisions", started, f"{agent.episode['steps']} steps")

stage, is_test, steps = agent.stage.name, agent.is_test, agent.episode["steps"]
agent.request_end_episode(W.FALL_PENALTY)   # what FallSensor does when a thigh or the pelvis touches the floor
run_until(lambda: any(r[0] == "done" for r in reports), 4 * ticks_per_decision)

times = [r for r in reports if r[0] == "time"]
dones = [r for r in reports if r[0] == "done"]
check("the fall is reported once", len(times) == 1 and len(dones) == 1, f"{len(times)} times, {len(dones)} results")
if times and dones:
    check("the report is the episode that fell: its length", abs(times[0][2] - steps * W.CONTROL_DT) < 1e-9,
          f"{times[0][2]:.2f} s reported, the episode ran {steps * W.CONTROL_DT:.2f} s")
    check("the report is the episode that fell: its stage and test flag", dones[0][1:] == (stage, is_test),
          f"reported {dones[0][1:]}, fell in {(stage, is_test)}")
check("the agent counted the fall", agent.falls == 1, f"{agent.falls} falls")

finish()
