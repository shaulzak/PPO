"""
The left/right mirror used by the symmetry loss: exact inverse of itself, and it matches the physics
(standing on the left foot looks like the mirror image of standing on the right foot).
    python tests/test_symmetry.py
"""
import numpy as np
import torch

from _common import check, finish, make_world

from bereshit import Vector3
from bereshit.addons.PPO import Academy, Trainer
import robot as R
import motions as M
import WalkAgent as W

print("test_symmetry")

x = torch.randn(7, W.OBS_DIM)
check("mirror(mirror(observation)) == observation", torch.allclose(W.mirror_observation(W.mirror_observation(x)), x))
a = torch.randn(7, W.ACTION_DIM)
check("mirror(mirror(action)) == action", torch.allclose(W.mirror_action(W.mirror_action(a)), a))
heading = torch.zeros(W.OBS_DIM)
heading[-2:] = torch.tensor([0.5, 0.866])          # turned 30 deg to the left since the start
check("the mirror image of turning left is turning right (heading: last two observations)",
      torch.allclose(W.mirror_observation(heading)[-2:], torch.tensor([-0.5, 0.866])))


def zero_act(self, observation, deterministic=False, **kwargs):
    return np.zeros(W.ACTION_DIM, dtype=np.float32), None, {
        "value": torch.tensor(0.0), "log_prob": torch.tensor(0.0), "policy_version": 0,
        "observation": torch.zeros(W.OBS_DIM), "continuous_action": torch.zeros(W.ACTION_DIM)}


Trainer.act = zero_act
Academy.setup_trainer(W.make_config(None))
Academy.get_trainer().inference_only = True


def observation_during_hold(first_leg):
    """Observation at the end of the first weight shift (the weight over one foot, just before the lift), zero
    corrections, no pushes. Not later: open loop the robot starts rolling off the standing foot's edge as the
    other foot lifts (standing on one foot needs the policy's balance feedback), so foot corners sit at the
    contact threshold and one run may count a corner as touching while the other doesn't."""
    root, robot = R.build_robot()
    floor = R.build_floor()
    agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage="one_leg", pushes=False, steady_pauses=False)
    root.add_component(agent)
    world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
    agent.first_leg = first_leg
    agent.clock = 0.0
    for step in range(int((M.ONE_LEG_PAUSE + M.ONE_LEG_LEAN - 0.05) / W.TICK)):
        world.update(step % 4 == 0)
    corners = agent._foot_corners()
    heights = np.array([c[1] for c in corners], dtype=np.float32)
    contacts = (heights < 0.003).astype(np.float32)
    return agent.sense(contacts, agent._reference(agent.motion_time)), heights


def mirrored_when(first_leg):
    root, robot = R.build_robot()
    agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage="one_leg", pushes=False, steady_pauses=False)
    agent.stage, agent.first_leg, agent.only_stance, agent.episode_time = W.STAGES[2], first_leg, None, 0.5
    agent.motion_time = 0.5
    return agent._mirrored()


check("one-leg stage: the network sees the mirror image while standing on the left foot",
      mirrored_when(1) and not mirrored_when(0))

left_up, left_heights = observation_during_hold(0)     # the weight on the right foot, the left about to lift
right_up, right_heights = observation_during_hold(1)   # and the other way round
diff = np.abs(W.mirror_observation(torch.tensor(left_up)).numpy() - right_up)
# the gait clock isn't used in this stage and runs the same in both runs: leave it out of the comparison
clock = slice(W.OBS_DIM - len(M.TASKS) - 5, W.OBS_DIM - len(M.TASKS) - 3)   # before: stage, speed, heading
diff[clock] = 0.0
# Foot contacts are on/off at 3 mm: the unloaded foot's corners sit right at it, so compare the corner
# heights themselves (mirrored the same way as the contact flags) instead of the flags.
contact_slots = slice(clock.start - 8, clock.start)
diff[contact_slots] = 0.0
as_obs = np.zeros(W.OBS_DIM, dtype=np.float32)
as_obs[contact_slots] = left_heights
mirrored_heights = W.mirror_observation(torch.tensor(as_obs)).numpy()[contact_slots]
height_diff = np.abs(mirrored_heights - right_heights).max()
check("standing on the left foot == mirror of standing on the right foot", diff.max() < 0.05,
      f"largest difference {diff.max():.3f} at observation index {int(diff.argmax())}")
check("foot corner heights mirror too (within 1 mm)", height_diff < 0.001, f"{height_diff * 1000:.2f} mm")


def pelvis_through_lift(first_leg):
    """Pelvis sideways position and roll every 0.1 s from the start through 60% of the first lift, zero
    corrections (at the end of the lift, open loop, the robot starts rolling off the standing foot's edge and
    any rounding difference grows there). The lift was where the legs parted: the engine solved a foot's corners one after another (the
    first one took the whole landing) and the reference hiked the left stance's hip the wrong way - 2.6 cm and
    6 deg apart. Both fixed, the runs stay within ~0.3 mm; with the heavier STS3095 hip-roll blocks single
    samples reach ~0.5 mm without growing (1 mm limit: still 25x below the old bug)."""
    root, robot = R.build_robot()
    floor = R.build_floor()
    agent = W.WalkAgent(robot, W.CONTROL_DT, fixed_stage="one_leg", pushes=False, steady_pauses=False)
    root.add_component(agent)
    world = make_world([root, floor], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
    agent.first_leg = first_leg
    agent.clock = 0.0
    samples = []
    for step in range(int((M.ONE_LEG_PAUSE + M.ONE_LEG_LEAN + 0.6 * M.ONE_LEG_LIFT) / W.TICK)):
        world.update(step % 4 == 0)
        if step % round(0.1 / W.TICK) == 0:
            pelvis = robot["pelvis"].transform
            up = R.local_to_world(pelvis.quaternion, Vector3(0, 1, 0))
            samples.append((pelvis.position.z, float(np.degrees(np.arcsin(np.clip(up.z, -1, 1))))))
    return np.array(samples)


left_lifts, right_lifts = pelvis_through_lift(0), pelvis_through_lift(1)
side_gap = np.abs(left_lifts[:, 0] + right_lifts[:, 0]).max()     # mirrored: z and roll change sign
roll_gap = np.abs(left_lifts[:, 1] + right_lifts[:, 1]).max()
check("lifting the left foot == mirror of lifting the right foot (pelvis within 1 mm, 0.1 deg)",
      side_gap < 0.001 and roll_gap < 0.1, f"{side_gap * 1000:.2f} mm, {roll_gap:.2f} deg")

finish()
