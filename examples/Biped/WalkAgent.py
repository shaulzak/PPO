"""
The biped's learning agent: one network that goes through the stages in curriculum.py.

Observations use only what the real robot can measure (IMU on the pelvis, the servo targets it sent,
the angles the servos report if they have feedback, 4 contact switches per foot), plus what it was asked
to do: the stage, the reference motion's targets, a gait clock and the walking speed.
Rewards may use anything the simulation knows (true velocity, center of mass...).
"""
import math
import random

import numpy as np
import torch

from bereshit import Vector3, Component
from bereshit.addons.PPO import Agent, Config

import robot as R
import motions as M
from curriculum import Curriculum, STAGES
from servos import SERVO
from walk_metrics import WalkStats, support_margin, capture_point

# Shared by Train.py, Run.py, Evaluate.py (and later the robot's controller): must match exactly.
TICK = 1 / 200           # physics step
CONTROL_DT = 1 / 50      # one decision every 4 physics steps
PHYSICS_EPOCHS = 30      # solver iterations per step
# The one-leg stage's motion waits in each pause until the robot stands well (see standing_well), so its
# episode can run longer than its motion: it ends when the motion is done, or at the latest here.
ONE_LEG_MAX_SECONDS = 60.0
MAX_EPISODE_SECONDS = max([s.seconds for s in STAGES] + [ONE_LEG_MAX_SECONDS])

FALL_HEIGHT = 0.5                 # pelvis center below this (m) = fallen
FALL_TILT_DEG = 40.0
FALL_PENALTY = -10.0              # big enough that lunging forward and falling doesn't pay
SWING_CLEARANCE = 0.02            # m: a swinging foot counts fully as lifted at this height
HOLD_CLEARANCE = 0.03             # m: the lifted foot in the one-leg stage (full reward from here up)
HOLD_COUNTS_FROM = 0.02           # m: the lifted foot's lowest corner, for the pass test's time on one foot
ONE_LEG_SETTLE = 0.5              # s into each one-leg pause before standing still is measured
BALANCE_FULL_MARGIN = 0.03        # m inside the support area counts as fully stable
LIMIT_MARGIN_DEG = 2.0            # a joint this close to its limit counts as "at the limit"
DRIFT_TIME = 1.2                  # s: velocities averaged this long (a gait cycle) for drift / sideways terms
MIRROR_TO_ONE_SIDE = {"one_leg"}  # stages where the network always sees the same standing side (see _mirrored)
SHIFT_M_PER_DEG = 0.006           # measured: 12 deg of lean moves the center of mass 7.2 cm, 19 deg 12 cm
PUSH_EVERY = (2.0, 4.0)           # s between pushes

N = R.NUM_SERVOS
FEEDBACK_DIM = N if SERVO.position_feedback else 0
# Last: the heading relative to the episode's start (sin, cos; on the robot: the IMU's yaw since the start).
OBS_DIM = 3 + 3 + N + FEEDBACK_DIM + N + N + 8 + 2 + len(M.TASKS) + 1 + 2
ACTION_DIM = N
RESIDUAL_SPAN_DEG = 20.0          # the most an action can move a joint away from the reference


# ----- left/right mirror (for the symmetry loss: what is learned on one leg carries to the other) -----
def _mirror_layout():
    """Index permutation and sign flips that turn an observation into its left/right mirror image."""
    perm, sign = [], []

    def same(n, signs=None):
        start = len(perm)
        perm.extend(range(start, start + n))
        sign.extend(signs or [1.0] * n)

    def swap_legs(n):
        start, half = len(perm), n // 2
        perm.extend(list(range(start + half, start + n)) + list(range(start, start + half)))
        sign.extend([1.0] * n)   # joint angles use a mirrored (anatomical) convention: no sign change

    same(3, [1.0, 1.0, -1.0])        # gravity in the pelvis frame: the sideways component flips
    same(3, [-1.0, -1.0, 1.0])       # angular velocity (axial vector): roll and yaw flip, pitch doesn't
    swap_legs(N)                     # servo targets sent
    if FEEDBACK_DIM:
        swap_legs(N)                 # servo angles reported
    swap_legs(N)                     # previous action
    swap_legs(N)                     # reference targets
    start = len(perm)                # foot corners: left <-> right, and inner <-> outer within a foot
    corner = [1, 0, 3, 2]
    perm.extend([start + 4 + corner[k] for k in range(4)] + [start + corner[k] for k in range(4)])
    sign.extend([1.0] * 8)
    same(2, [-1.0, -1.0])            # gait clock: the other leg's phase = half a cycle later
    same(len(M.TASKS) + 1)           # stage and command speed
    same(2, [-1.0, 1.0])             # heading: turned left <-> turned right
    assert len(perm) == OBS_DIM
    return torch.tensor(perm), torch.tensor(sign)


_OBS_PERM, _OBS_SIGN = _mirror_layout()
_ACT_PERM = torch.tensor(list(range(N // 2, N)) + list(range(N // 2)))


def mirror_observation(obs):
    """obs: tensor or numpy array [..., OBS_DIM] (raw, not normalized)."""
    if isinstance(obs, np.ndarray):
        return (obs[..., _OBS_PERM.numpy()] * _OBS_SIGN.numpy()).astype(obs.dtype)
    return obs[..., _OBS_PERM] * _OBS_SIGN.to(obs.dtype)


def mirror_action(action):
    if isinstance(action, np.ndarray):
        return action[..., _ACT_PERM.numpy()]
    return action[..., _ACT_PERM]


def check_model_servo(path):
    """Warn when a checkpoint was trained for a different robot configuration than the current one."""
    checkpoint = torch.load(path, map_location="cpu")
    meta = checkpoint.get("config", {}).get("metadata") or {}
    if not meta:
        print(f"Note: {path} doesn't record its robot configuration (trained before it was saved).")
        return
    current = model_metadata()
    for key, value in current.items():
        if key in meta and meta[key] != value:
            print(f"WARNING: {path} was trained with {key}={meta[key]}, but the current configuration has {value}.")


def model_metadata():
    return {"servo": SERVO.name, "obs_dim": OBS_DIM, "hip_yaw": R.HIP_YAW, "leg_channel": R.LEG_CHANNEL,
            "battery": R.BATTERY}


def make_config(best_model_path):
    return Config(
        metadata=model_metadata(),
        obs_dim=OBS_DIM,
        action_dim_continuous=ACTION_DIM,
        hidden_size=128,          # ~70 -> 128 -> 128 -> 12: small enough for an ESP32
        rollout_steps=4096,
        minibatch_size=256,
        update_epochs=5,
        # 1e-4 (was 3e-4): late in the one-leg stage the results swung (right foot survived 1.00 -> 0.75 within
        # ~40 min) - fine-tuning a policy that already works needs smaller steps.
        learning_rate=1e-4,
        gamma=0.995,              # ~4 s ahead at 50 decisions/s (0.99 made falls 2.5 s later nearly free)
        # 0.001: at 0.005 the bonus held the noise at std ~0.35 for hours (rewards are ~0.2 per step), the
        # noisy robot couldn't follow the weight shift, and the noise-free tests drifted (0.93 -> 0.5).
        entropy_coef=0.001,
        initial_log_std=-1.0,     # start exploring with std ~0.37 (~+-7 deg around the reference)
        separate_value_network=True,
        # Start by playing the reference motion exactly: stand and shift are stable on their own, and a
        # policy that began with big corrections kept them as a habit (and fell in the shift stage).
        zero_initial_actions=True,
        # The policy must mirror itself between the legs (what it learns standing on the right foot it
        # also does on the left): it stood noticeably better on one leg than the other.
        symmetry_weight=0.5,
        max_steps=int(MAX_EPISODE_SECONDS / CONTROL_DT) + 10,   # the agent ends episodes per stage itself
        best_model_path=best_model_path,
    )


def combine_reward(parts):
    """One step's reward from its weighted terms: the positive values add up, and the negative ones (the
    penalties) shrink the "alive" bonus instead of being subtracted:

        reward = sum of positive terms + alive * exp(-sum of penalties)

    Never below zero, so living on always beats falling (with a plain sum the average step went negative,
    and a fall that ends the episode cost less than living on: the policy learned to fall). A hard floor at
    zero fixed that but cut 40% of the one-leg steps to exactly 0, where "almost lifted" and "never tried"
    looked the same. Here every penalty always counts. Returns (reward, share of the alive bonus kept)."""
    positive = sum(v for k, v in parts.items() if k != "alive" and v > 0)
    penalty = -sum(v for k, v in parts.items() if k != "alive" and v < 0)
    kept = math.exp(-penalty)
    return positive + parts.get("alive", 0.0) * kept, kept


def facing_term(degrees_off):
    """Reward for facing the start direction: +1 straight ahead, down to 0 at 45 deg, and a penalty only
    beyond that (-1 at 90). As a penalty from the start (1 at 20 deg, up to 3) it shrank the alive bonus: at
    34 deg only a fifth of it was left, so falling cost almost nothing and turning further barely mattered -
    survival dropped and the turning grew. As a bonus it pulls back at every angle and never makes falling
    cheap."""
    return max(-1.0, 1.0 - abs(degrees_off) / 45.0)


def stillness_term(flat_share, pelvis_speed, tilt_rate=0.0):
    """Reward for standing still on both feet (the one-leg pauses, the stand stage between pushes): the share
    of foot corners on the floor times how still the pelvis is - 1 flat and not moving, 0 from 0.1 m/s - times
    how little it tilts (pitch/roll rate, rad/s; 0 from 1 rad/s = 57 deg/s). The old one (+0.5 with any corner
    of each foot down, minus the speed) paid for rocking from foot edge to foot edge at ~5 Hz, which the policy
    learned (still_stable 0.22, and the feet slid round: turning in place). Without the tilt factor the feet
    went flat but a ~5 Hz tremor stayed (1-2 mm, ~1 deg; pelvis tilting ~18 deg/s, moving 5-6 cm/s: still_stable
    stuck ~0.45). The tilt rate, not the change of the commands: in training that is mostly exploration noise."""
    return flat_share * max(0.0, 1.0 - pelvis_speed / 0.1) * max(0.0, 1.0 - tilt_rate / 1.0)


def quiet_motors_term(motor_speeds):
    """Reward for motors that don't move while the robot should stand still: from the average speed of the
    motors (deg/s), +1 not moving, 0 at 20 deg/s, down to -1 from 40 deg/s - a straight line, every deg/s
    less pays 0.05 more. The ~5 Hz tremor (about 1 deg back and forth) is ~20 deg/s: too small at the pelvis
    (1-2 mm) for stillness_term to see it well, plain in the motors. Floored at -1 so it never eats the alive
    bonus (see facing_term). Tried at weight 1.0 in the one-leg pauses and the stand stage: made it WORSE (motors
    17.8 -> 28.2 deg/s, still_stable unchanged ~0.3) - in training the exploration noise itself moves the motors,
    so it sat near -1 everywhere and gave no direction. Not used by any stage now; kept for probe_still."""
    return max(-1.0, 1.0 - float(np.mean(np.abs(motor_speeds))) / 20.0)


# The one-leg pauses are event-driven (the user's rule): from ONE_LEG_SETTLE s into a pause the motion waits
# until the robot has stood well for curriculum.steady_hold s in a row, then goes on to the weight shift.
# "Standing well": every foot corner on the floor, the capture point inside the feet, no tremor (pelvis slower
# than the still_stable limit, hardly tilting), not turning. Not "motors quiet": in training the exploration
# noise alone moves the motors ~39 deg/s, so it failed every moment (and a quiet-motors reward had made the
# tremor worse).
STEADY_SPEED = 0.05               # m/s, pelvis (the still_stable limit)
STEADY_TILT_DEG = 20.0            # deg/s, pelvis pitch/roll rate
STEADY_YAW_DEG = 10.0             # deg/s, pelvis turning rate
# While waiting, the only reward is the streak: STEADY_POINT per step of standing well, and a break takes the
# whole streak back (the sum returns to what the earlier phases banked). No alive bonus there: waiting pays
# nothing, only moving on to the next phases does - hiding in the easy pause never pays.
STEADY_POINT = 3.0                # ~ a normal step's reward
# Exploration noise while it should stand still (the one-leg pauses, the stand stage between pushes), instead
# of the learned ~0.15 (the user's call, only there): without noise the policy stood well up to 0.98 s in a
# row, with noise 0.03 only 0.18 s, 0.01 0.86 s, 0.15 never - the noise itself was the tremor it could never
# learn away. The lifting keeps the full noise to go on learning. The real robot has no noise at all.
STILL_NOISE_STD = 0.01


def standing_well(contacts, icp_margin, pelvis_speed, tilt_rate, yaw_rate):
    """True when this moment of a one-leg pause counts toward the steady streak (rates in rad/s)."""
    return (min(contacts) > 0 and icp_margin > 0.0 and pelvis_speed < STEADY_SPEED
            and math.degrees(tilt_rate) < STEADY_TILT_DEG and math.degrees(abs(yaw_rate)) < STEADY_YAW_DEG)


class FallSensor(Component):
    """On body parts that must never touch the floor (everything above the ankles)."""

    def __init__(self, agent):
        super(FallSensor, self).__init__()
        self.agent = agent

    def OnCollisionEnter(self, collision):
        if collision.other.parent.get_component("Floor"):
            self.agent.request_end_episode(FALL_PENALTY)


def _vec(v):
    return np.array([v.x, v.y, v.z])


class WalkAgent(Agent):
    def __init__(self, robot, control_dt, curriculum=None, fixed_stage=None, pushes=True, steady_pauses=True):
        """steady_pauses=False: the one-leg pauses keep their fixed length (tests of the reference motion
        alone, e.g. hanging from the pelvis, where the robot can't stand well)."""
        super(WalkAgent, self).__init__()
        self.steady_pauses = steady_pauses
        self.robot = robot
        self.control_dt = control_dt
        self.curriculum = curriculum or Curriculum()
        self.fixed_stage = next((s for s in STAGES if s.name == fixed_stage), None) if fixed_stage else None
        self.pushes = pushes
        self.pelvis = robot["pelvis"]
        self.servos = robot["servos"]
        self.feet = [robot["left"]["foot"], robot["right"]["foot"]]
        self.bodies = [robot["pelvis"]] + list(robot["left"].values()) + list(robot["right"].values())
        self.masses = np.array([b.Rigidbody.mass for b in self.bodies])
        self.total_mass = float(self.masses.sum())
        self.nominal_height = R.axis_heights()["pelvis"]
        self.limits = [(s.low, s.high) for s in self.servos]
        self.rated_share = SERVO.rated_torque / SERVO.stall_torque
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.action_before_previous = np.zeros(ACTION_DIM, dtype=np.float32)
        self.stage = STAGES[0]
        self.is_test = False
        self.command_speed = 0.0
        self.clock = 0.0
        self.episode_time = 0.0
        self.motion_time = 0.0            # the reference motion's clock (one-leg: stops while a pause waits)
        self.falls = 0
        self.episodes = 0
        self.stats = WalkStats()
        self.pushes_given = 0
        self.start_yaw = 0.0

    def _yaw(self):
        """Pelvis heading about the vertical, rad (0 = facing +x, positive toward +z = the robot's left)."""
        forward = R.local_to_world(self.pelvis.transform.quaternion, Vector3(1, 0, 0))
        return math.atan2(forward.z, forward.x)

    def attach(self, parent):
        for side in ("left", "right"):
            for name in ("thigh", "shin", "hip_block"):
                self.robot[side][name].add_component(FallSensor(self))
        self.pelvis.add_component(FallSensor(self))

    # ----- episode -----
    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        if self.fixed_stage:
            self.stage, self.is_test = self.fixed_stage, False
        else:
            # A test episode runs like Run.py: best action, nothing stored, no learning. Only test
            # episodes decide whether a stage is passed. (Setting the flag directly rather than through
            # set_inference_mode(), which would also throw away the rollout collected so far.)
            self.stage, self.is_test = self.curriculum.choose()
            self.trainer.inference_only = self.is_test
        self.command_speed = random.uniform(*self.stage.command_range)
        # Gait clock starts near a double-support moment, with either leg about to swing.
        self.clock = random.choice((0.0, 0.5)) * M.GAIT_PERIOD + random.uniform(-0.03, 0.03)
        self.episode_time = 0.0
        self.motion_time = 0.0
        self.steady_steps = 0             # current streak of standing well in a one-leg pause, and its points
        self.steady_points = 0.0
        self.waited = 0.0                 # s this pause has waited so far
        self.standing_well_now = False
        self.velocity_average = np.zeros(3)
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.action_before_previous = np.zeros(ACTION_DIM, dtype=np.float32)
        self.previous_angles = np.array([s.angle() for s in self.servos])
        self.feet_down = [True, True]
        self.next_push = random.uniform(*PUSH_EVERY) + 1.0
        self.last_push = -1e9
        com = self._center_of_mass()
        self.com_start_x, self.com_start_z = com[0], com[2]
        self.start_yaw = self._yaw()
        self.feet_start = [_vec(f.transform.position) for f in self.feet]
        # Which leg lifts first in the one-leg stage: random, so both legs get the same practice (the left
        # always lifting first, right after the start, was one reason the robot stood better on the right).
        self.first_leg = random.choice((0, 1))
        # Once one leg passes the one-leg test and the other doesn't, training episodes stand only on the
        # weaker one (tests keep checking both; if the good leg slips, both are practised again).
        self.only_stance = None
        if self.stage.name == "one_leg" and not self.is_test:
            self.only_stance = self.curriculum.weaker_leg()
        self.fall_stance = None
        self.episode = {"steps": 0, "shift_track": 0.0, "hold_steps": [0, 0], "hold_single": [0, 0], "single": 0,
                        "single_runs": 0, "single_side": None, "speed_ratio": 0.0,
                        "still_steps": 0, "still_ok": 0, "lift_postures": []}
        self._apply(self._reference(0.0), self.previous_action)
        self.episodes += 1

    def _finish(self, terminated, survived, final_observation=None):
        """End the episode, report it to the curriculum, and start the next one."""
        if self._waiting() and self.waited > 0.0 and not self.is_test:
            self.curriculum.steady_pause_done(False)      # ended (fell / time up) before standing well
        e = self.episode
        steps = max(1, e["steps"])
        self.curriculum.add_time(self.stage.name, e["steps"] * self.control_dt)
        self.curriculum.episode_done(self.stage.name, {
            "survived": float(survived),
            "shift_track": e["shift_track"] / steps,
            # share of the hold time spent standing alone on that foot (the other one lifted)
            "on_left_foot": e["hold_single"][0] / max(1, e["hold_steps"][0]),
            "on_right_foot": e["hold_single"][1] / max(1, e["hold_steps"][1]),
            # per standing foot: 0 = fell standing on it, 1 = stood on it and didn't fall there, None = the
            # episode never got to it (left out of that foot's average)
            "survived_on_left": self._survived_on(0, survived),
            "survived_on_right": self._survived_on(1, survived),
            "single_stretch": e["single"] / max(1, e["single_runs"]) * self.control_dt,
            "single_fraction": e["single"] / steps,
            "speed_ratio": e["speed_ratio"] / steps,
            # stand / one-leg stages: share of the "standing still" moments spent stable on both feet
            "still_stable": e["still_ok"] / max(1, e["still_steps"]),
        }, self.is_test)
        if self.has_active_action:
            self.end_episode(terminated=terminated, final_observation=final_observation)
        else:
            # end_episode() does nothing before the first action (nothing to close): reset directly.
            self.begin_episode()

    # ----- sensing -----
    def _gravity_in_pelvis(self):
        g = R.world_to_local(self.pelvis.transform.quaternion, Vector3(0, -1, 0))
        return np.array([g.x, g.y, g.z], dtype=np.float32)

    def _foot_corners(self):
        """World (x, y, z) of the 4 bottom corners of each foot (left first)."""
        corners = []
        hx, hy, hz = R.FOOT_LENGTH / 2, R.FOOT_THICKNESS / 2, R.FOOT_WIDTH / 2
        for foot in self.feet:
            q, p = foot.transform.quaternion, foot.transform.position
            for cx, cz in ((hx, hz), (hx, -hz), (-hx, hz), (-hx, -hz)):
                c = p + R.local_to_world(q, Vector3(cx, -hy, cz))
                corners.append((c.x, c.y - R.FLOOR_TOP, c.z))
        return corners

    def _center_of_mass(self):
        p = np.array([_vec(b.transform.position) for b in self.bodies])
        return (self.masses[:, None] * p).sum(0) / self.total_mass

    def _com_velocity(self):
        v = np.array([_vec(b.Rigidbody.velocity) for b in self.bodies])
        return (self.masses[:, None] * v).sum(0) / self.total_mass

    def _phase(self, t=None):
        clock = self.clock + ((t - self.motion_time) if t is not None else 0.0)
        return (clock % M.GAIT_PERIOD) / M.GAIT_PERIOD

    def _reference(self, t):
        """Targets at motion time t."""
        return M.reference_angles(self.stage.name, t, self._phase(t), self.command_speed, self.first_leg,
                                  self.only_stance)

    def _one_leg(self, t=None):
        """(lean, lifted side, lift 0..1) of the one-leg schedule now (or at t)."""
        return M.one_leg_schedule(self.motion_time if t is None else t, self.first_leg, self.only_stance)

    def _waiting(self):
        """One-leg stage: the pause is waiting for the robot to stand well (from ONE_LEG_SETTLE s in)."""
        return (self.steady_pauses and self.stage.name == "one_leg"
                and M.one_leg_still(self.motion_time, ONE_LEG_SETTLE))

    def _steady_step(self):
        """One step of a waiting pause: the streak reward, and the motion goes on once the streak is long enough."""
        self.waited += self.control_dt
        if not self.standing_well_now:
            reward = -self.steady_points          # the streak broke: back to what was banked before it
            self.steady_steps, self.steady_points = 0, 0.0
            return reward
        self.steady_steps += 1
        self.steady_points += STEADY_POINT
        if self.steady_steps >= max(1, round(self.curriculum.steady_hold / self.control_dt)):
            if not self.is_test:
                self.curriculum.steady_pause_done(self.waited <= M.ONE_LEG_PAUSE - ONE_LEG_SETTLE + 1e-9)
            half = M.ONE_LEG_CYCLE / 2
            self.motion_time = self.motion_time - self.motion_time % half + M.ONE_LEG_PAUSE   # on to the shift
            self.steady_steps, self.steady_points, self.waited = 0, 0.0, 0.0
        return STEADY_POINT

    def _mirrored(self):
        """One-leg stage: the network always stands on the right foot (the left one lifts). Standing on the
        left, it gets the mirror image and its command is mirrored back, so the two legs share one skill:
        what it learns on one leg is learned on the other (the symmetry loss alone left a gap)."""
        return self.stage.name in MIRROR_TO_ONE_SIDE and self._one_leg()[1] == 1

    def sense(self, contacts, reference):
        """The observation vector, in the same order the robot's controller must build it."""
        w = R.world_to_local(self.pelvis.transform.quaternion, self.pelvis.Rigidbody.angular_velocity)
        phase = 2 * math.pi * self._phase()
        feedback = [s.angle() / 90.0 for s in self.servos] if SERVO.position_feedback else []
        task = [1.0 if name == self.stage.name else 0.0 for name in M.TASKS]
        # Without it the policy could feel itself turning (the yaw rate) but not how far it had turned, and
        # couldn't turn back: it stood 47 deg off after learning the one-leg stage.
        yaw = self._yaw() - self.start_yaw
        return np.concatenate([
            self._gravity_in_pelvis(), [w.x, w.y, w.z],
            [s.command_deg / 90.0 for s in self.servos], feedback, self.previous_action,
            np.asarray(reference) / 90.0, contacts,
            [math.sin(phase), math.cos(phase)], task, [self.command_speed],
            [math.sin(yaw), math.cos(yaw)],
        ]).astype(np.float32)

    # ----- reward (simulation-only information is fine here) -----
    def _reward(self, corners, contacts, reference):
        """Reward for the state reached by the previous action. Returns (parts, metrics)."""
        stage, w = self.stage, self.stage.weights
        q = self.pelvis.transform.quaternion
        v_pelvis = _vec(self.pelvis.Rigidbody.velocity)
        forward = R.local_to_world(q, Vector3(1, 0, 0))
        f = np.array([forward.x, forward.z])
        f = f / (np.linalg.norm(f) + 1e-6)
        v_forward = v_pelvis[0] * f[0] + v_pelvis[2] * f[1]
        v_side = -v_pelvis[0] * f[1] + v_pelvis[2] * f[0]
        # Feet in the pelvis frame: x = how far the left foot is ahead of the right, z = how far apart.
        feet = R.world_to_local(q, self.feet[0].transform.position - self.feet[1].transform.position)
        k = min(1.0, self.control_dt / DRIFT_TIME)
        self.velocity_average = (1 - k) * self.velocity_average + k * v_pelvis
        avg_side = -self.velocity_average[0] * f[1] + self.velocity_average[2] * f[0]
        g = self._gravity_in_pelvis()
        height = self.pelvis.transform.position.y
        heights = np.array([c[1] for c in corners])
        left_down, right_down = contacts[:4].max() > 0, contacts[4:].max() > 0
        down = [left_down, right_down]

        com = self._center_of_mass()
        com_v = self._com_velocity()
        touching = [(c[0], c[2]) for c, d in zip(corners, contacts) if d > 0]
        com_margin = support_margin((com[0], com[2]), touching)
        icp = capture_point((com[0], com[2]), (com_v[0], com_v[2]), com[1])
        icp_margin = support_margin(icp, touching)

        angles = np.array([s.angle() for s in self.servos])
        imitation = math.exp(-float(np.mean(((angles - np.asarray(reference)) / 15.0) ** 2)))
        torque_use = np.array([abs(s.joint.motor_impulse / TICK) / s.max_torque for s in self.servos])
        # Per motor, to see that every motor of both legs does its job: how far it is from the angle it was
        # told to go to, and how much it moves.
        tracking_error = np.abs(angles - np.array([s.command_deg for s in self.servos]))
        movement = np.abs(angles - self.previous_angles) / self.control_dt
        self.previous_angles = angles
        at_limit = float(np.mean([a < lo + LIMIT_MARGIN_DEG or a > hi - LIMIT_MARGIN_DEG
                                  for a, (lo, hi) in zip(angles, self.limits)]))

        slip, impact = 0.0, 0.0
        for i, foot in enumerate(self.feet):
            fv = foot.Rigidbody.velocity
            if down[i]:
                slip += math.hypot(fv.x, fv.z)
                if not self.feet_down[i]:           # touchdown
                    impact = max(impact, max(0.0, -fv.y))
                    rel = foot.transform.position - self.feet[1 - i].transform.position
                    self.stats.record_touchdown(abs(rel.x * f[0] + rel.z * f[1]), max(0.0, -fv.y))
        self.feet_down = down

        # ----- stage task terms -----
        e = self.episode
        task = 0.0
        pelvis_speed = math.hypot(v_pelvis[0], v_pelvis[2])
        av = self.pelvis.Rigidbody.angular_velocity
        tilt_rate = math.hypot(av.x, av.z)          # pitch/roll, not yaw
        # Where the reference motion expects the center of mass: the start spot, moved sideways by the lean
        # in the shift / one-leg stages. Drift is measured from there, by position: a velocity average
        # counted the 3 s weight-shift sway itself as drift and taught the policy to fight it (and fall).
        lean_now = 0.0
        if stage.name == "shift":
            lean_now = M.SHIFT_DEG * math.sin(2 * math.pi * self.motion_time / M.SHIFT_PERIOD)
        elif stage.name == "one_leg":
            lean_now = self._one_leg()[0]
        expected_z = self.com_start_z - SHIFT_M_PER_DEG * lean_now
        drift_distance = math.hypot(com[0] - self.com_start_x, com[2] - expected_z)
        if stage.name == "shift":
            task = math.exp(-((com[2] - expected_z) / 0.03) ** 2)
            e["shift_track"] += task
        elif stage.name == "one_leg":
            _, lifted, lift = self._one_leg()
            if lift > 0.5:
                swing, stance = lifted, 1 - lifted
                lifted_share = min(1.0, max(0.0, heights[4 * swing:4 * swing + 4].min() / HOLD_CLEARANCE))
                # How steady it is on the standing foot: 1 with the capture point >= 3 cm inside the foot,
                # 0.5 at its edge, 0 once it is 3 cm outside (starting to fall, before any fall happens).
                steady = 0.5 * (1.0 + max(-1.0, min(1.0, icp_margin / BALANCE_FULL_MARGIN)))
                # Only lifting pays here (keeping both feet down scored as well as lifting and the policy
                # never lifted), and it pays by how steady the robot is while the foot is up. From 0, not -1:
                # below-zero steps are cut to zero by the reward floor, which hid "almost lifted" from
                # "never tried" (the average step had gone below zero).
                task = lifted_share * steady if down[stance] else 0.0
                e["hold_steps"][stance] += 1
                # Counts as standing on one foot only with the other one clearly up, not hovering a few mm.
                e["hold_single"][stance] += int(down[stance] and
                                                heights[4 * swing:4 * swing + 4].min() >= HOLD_COUNTS_FROM)
                if not self.is_test:
                    self._record_hold(stance, swing, heights, down, com, icp_margin, g, torque_use, tracking_error)
            elif M.one_leg_still(self.motion_time):
                task = stillness_term(float(contacts.mean()), pelvis_speed, tilt_rate)
            else:
                task = 0.5 if (left_down and right_down) else -0.5
        elif stage.name == "stand" and min(self.episode_time, self.episode_time - self.last_push) > 1.0:
            task = stillness_term(float(contacts.mean()), pelvis_speed, tilt_rate)
        # "Standing still" moments, measured for the pass test: in the stand stage from 1 s after the start
        # and after each push, in the one-leg stage the pauses before each weight shift, from 0.5 s in: the
        # pause starts right as the weight comes back to the middle (or as the robot lands at the start), and
        # the first 0.3-0.6 s is that motion dying out, even with feet flat and still after it.
        still_window = ((stage.name == "stand" and min(self.episode_time, self.episode_time - self.last_push) > 1.0)
                        or self._waiting())
        self.standing_well_now = standing_well(contacts, icp_margin, pelvis_speed, tilt_rate, av.y)
        if still_window:
            e["still_steps"] += 1
            e["still_ok"] += int(left_down and right_down and icp_margin > 0.0 and pelvis_speed < 0.05)
        gait = 0.0
        if stage.name in ("march", "walk"):
            scale = min(1.0, self.episode_time / M.GAIT_START_SECONDS)
            if scale > 0.3:                          # while the step is still growing from rest, no demand
                swings = M.gait_windows(self._phase())
                clearance = SWING_CLEARANCE * scale
                score = 0.0
                for i in (0, 1):
                    if swings[i]:
                        score += 2.0 * min(1.0, heights[4 * i:4 * i + 4].min() / clearance) - 1.0
                    elif swings[1 - i]:
                        score += 0.5 if down[i] else -0.5
                gait = 0.5 * score
        speed = 0.0
        if stage.name == "walk":
            cmd = self.command_speed
            speed = 1.5 * max(-1.0, min(v_forward, cmd) / cmd) - 2.0 * max(0.0, v_forward - 1.2 * cmd) / cmd
            e["speed_ratio"] += v_forward / cmd

        side = "L" if left_down and not right_down else "R" if right_down and not left_down else None
        if side:
            e["single"] += 1
            if side != e["single_side"]:
                e["single_runs"] += 1
        e["single_side"] = side
        e["steps"] += 1

        terms = {
            "alive": 1.0,
            "balance": max(-1.0, min(1.0, icp_margin / BALANCE_FULL_MARGIN)),
            "imitate": imitation,
            "task": task,
            "gait": gait,
            "speed": speed,
            "still": -math.hypot(v_pelvis[0], v_pelvis[2]),
            "drift": -min(1.0, drift_distance / 0.1),
            "side": -min(2.0, abs(avg_side) / max(self.command_speed, 0.03)),
            "heading": -(1.0 - f[0]),
            # Standing stages: the feet stay side by side (in the shift stage one foot crept forward).
            "stagger": -min(1.0, abs(feet.x) / 0.05),
            # Standing stages: keep facing the start direction ("heading" is ~0.06 at 20 deg: too weak).
            "facing": facing_term(math.degrees(math.atan2(f[1], f[0]))),
            # Only while it should stand still on both feet (the same moments still_stable measures).
            "quiet": quiet_motors_term(movement) if still_window else 0.0,
            "upright": -float(g[0] ** 2 + g[2] ** 2),
            "height": -(height - self.nominal_height) ** 2,
            "slip": -slip,
            "impact": -impact ** 2,
            "energy": -float(np.mean(torque_use ** 2)),
            "limit": -at_limit,
            "smooth": -float(np.mean((self.previous_action - self.action_before_previous) ** 2)),
            # Correct the reference only where it helps: unneeded corrections learned in one stage were
            # carried into the next one as a habit.
            "residual": -float(np.mean(self.previous_action ** 2)),
            "yaw": -self.pelvis.Rigidbody.angular_velocity.y ** 2,
        }
        parts = {name: weight * terms[name] for name, weight in w.items()}
        metrics = {
            "v_forward": v_forward, "v_side_abs": abs(v_side), "command": self.command_speed, "height": height,
            "roll": math.degrees(math.atan2(g[2], -g[1])), "pitch": math.degrees(math.atan2(g[0], -g[1])),
            "heading_error": abs(math.degrees(math.atan2(f[1], f[0]))),
            "stance_width": abs(feet.z), "feet_stagger": feet.x, "feet_stagger_abs": abs(feet.x),
            "icp_margin": icp_margin, "icp_inside": float(icp_margin > 0),
            "com_margin": com_margin, "com_inside": float(com_margin > 0),
            "slip": slip, "residual": float(np.mean(np.abs(self.previous_action))),
            **{f"torque_{j}": 0.5 * (torque_use[R.servo_index(j, 0)] + torque_use[R.servo_index(j, 1)])
               for j in R.JOINT_ORDER},
            "torque_peak": float(torque_use.max()), "over_rated": float(np.mean(torque_use > self.rated_share)),
            "at_limit": at_limit,
            "current": float(np.sum(SERVO.idle_current + SERVO.stall_current * torque_use)),
            **{f"motor_{kind}_{side}_{j}": float(values[R.servo_index(j, i)])
               for kind, values in (("torque", torque_use), ("error", tracking_error), ("moves", movement))
               for i, side in enumerate("LR") for j in R.JOINT_ORDER},
        }
        return parts, metrics, (left_down, right_down)

    # ----- termination -----
    def _simulation_blew_up(self):
        for b in self.bodies:
            p, v, a = b.transform.position, b.Rigidbody.velocity, b.Rigidbody.angular_velocity
            if not all(math.isfinite(x) and abs(x) < 100 for x in (p.x, p.y, p.z, v.x, v.y, v.z, a.x, a.y, a.z)):
                return True
        return False

    def _fall_reason(self):
        if self.pelvis.transform.position.y < FALL_HEIGHT:
            return "height"
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, -float(self._gravity_in_pelvis()[1])))))
        return "tilt" if tilt > FALL_TILT_DEG else None

    def _record_hold(self, stance, swing, heights, down, com, icp_margin, g, torque_use, tracking_error):
        """One step of holding a foot up, for the per-leg table in the report (standing on the left | right).
        Sideways values are measured toward the lifted foot, so both sides read the same way."""
        stance_foot, lifted_foot = self.feet[stance].transform.position, self.feet[swing].transform.position
        dx, dz = lifted_foot.x - stance_foot.x, lifted_foot.z - stance_foot.z
        n = max(1e-6, math.hypot(dx, dz))
        up = R.local_to_world(self.pelvis.transform.quaternion, Vector3(0, 1, 0))
        lift = float(heights[4 * swing:4 * swing + 4].min())
        joint = lambda name, side: R.servo_index(name, side)
        self.stats.record_hold("LR"[stance], {
            "alone": float(down[stance] and lift >= HOLD_COUNTS_FROM),
            "lift_cm": 100 * lift,
            "com_toward_lifted_cm": 100 * ((com[0] - stance_foot.x) * dx + (com[2] - stance_foot.z) * dz) / n,
            "icp_margin_cm": 100 * icp_margin,
            "icp_inside": float(icp_margin > 0),
            "tilt_toward_lifted_deg": math.degrees(math.asin(max(-1.0, min(1.0, (up.x * dx + up.z * dz) / n)))),
            "pitch_deg": math.degrees(math.atan2(g[0], -g[1])),
            "stance_hip_roll_torque": torque_use[joint("hip_roll", stance)],
            "stance_ankle_roll_torque": torque_use[joint("ankle_roll", stance)],
            "stance_ankle_pitch_torque": torque_use[joint("ankle_pitch", stance)],
            "stance_hip_roll_off_deg": tracking_error[joint("hip_roll", stance)],
            "stance_ankle_roll_off_deg": tracking_error[joint("ankle_roll", stance)],
            "lifted_knee_off_deg": tracking_error[joint("knee_flex", swing)],
            "lifted_hip_flex_off_deg": tracking_error[joint("hip_flex", swing)],
        })

    def _lift_label(self):
        """'first' for the episode's first lift, 'later' for the ones after it (the motion time's half cycle)."""
        return "first" if self.motion_time < M.ONE_LEG_CYCLE / 2 else "later"

    def _note_lift_start(self):
        """As a weight shift onto one foot begins (the pause before it is over): how the robot stands then.
        The first lift starts from the clean starting pose, the later ones from wherever the one before left
        the feet and the body, and the tests fell on the second lift much more than on the first."""
        half = M.ONE_LEG_CYCLE / 2
        index = int(self.motion_time // half)
        if self.motion_time % half < M.ONE_LEG_PAUSE or index < len(self.episode["lift_postures"]):
            return
        _, lifted, _ = self._one_leg()
        stance = 1 - lifted
        posture = self.lift_start_posture(stance)
        self.episode["lift_postures"].append(posture)
        if not self.is_test:
            self.stats.record_lift_start(self._lift_label(), posture)

    def lift_start_posture(self, stance):
        """Feet and center of mass now, seen from the foot about to be stood on (sideways + = toward the
        other foot, forward + = where the pelvis faces)."""
        q = self.pelvis.transform.quaternion
        stance_foot, other_foot = self.feet[stance].transform.position, self.feet[1 - stance].transform.position
        feet = R.world_to_local(q, other_foot - stance_foot)
        mid = (_vec(stance_foot) + _vec(other_foot)) / 2
        com = R.world_to_local(q, Vector3(*(self._center_of_mass() - mid)))
        side = 1.0 if feet.z >= 0 else -1.0                    # pelvis z toward the other foot
        def foot_yaw(foot):
            f = R.local_to_world(foot.transform.quaternion, Vector3(1, 0, 0))
            return math.atan2(f.z, f.x)
        yaw_gap = math.degrees(foot_yaw(self.feet[1 - stance]) - foot_yaw(self.feet[stance]))
        moved = [100 * math.hypot(*(_vec(f.transform.position) - s)[[0, 2]]) for f, s in zip(self.feet, self.feet_start)]
        return {
            "feet_apart_cm": 100 * abs(feet.z),
            "other_foot_ahead_cm": 100 * feet.x,
            "feet_turned_apart_deg": (yaw_gap + 180) % 360 - 180,
            "com_toward_other_cm": 100 * com.z * side,
            "com_ahead_cm": 100 * com.x,
            "pelvis_turned_deg": math.degrees((self._yaw() - self.start_yaw + math.pi) % (2 * math.pi) - math.pi),
            "stance_foot_moved_cm": moved[stance],
            "other_foot_moved_cm": moved[1 - stance],
        }

    def _survived_on(self, side, survived):
        if not survived and self.fall_stance == side:
            return 0.0
        return 1.0 if self.episode["hold_steps"][side] > 0 else None

    def _count_fall_phase(self):
        """One-leg stage: which part of the motion a fall happened in, and on which foot."""
        if self.stage.name != "one_leg":
            return
        _, lifted, lift = self._one_leg()
        if lift > 0.0:
            self.fall_stance = 1 - lifted              # fell while standing on this foot
        if self._one_leg()[0] != 0.0 or lift > 0.0:      # after this lift's weight shift began
            self.stats.count("lift_fall_" + self._lift_label())
        stance = "LR"[1 - lifted]
        if lift > 0.5:
            where = f"hold_{stance}"                  # holding the other foot up
        elif lift > 0.0:
            where = f"lift_{stance}"                  # lifting / putting down the other foot
        elif M.one_leg_still(self.motion_time):
            where = "still"
        else:
            where = "shift"
        self.stats.count("where_" + where)

    def _push(self):
        """A kick to the pelvis in a random horizontal direction (robustness; the push test in Evaluate)."""
        angle = random.uniform(0, 2 * math.pi)
        dv = random.uniform(0.3, 1.0) * self.stage.push_speed
        self.give_push(dv * math.cos(angle), dv * math.sin(angle))
        self.next_push = self.episode_time + random.uniform(*PUSH_EVERY)

    def give_push(self, dvx, dvz):
        v = self.pelvis.Rigidbody.velocity
        self.pelvis.Rigidbody.velocity = Vector3(v.x + dvx, v.y, v.z + dvz)
        self.pushes_given += 1
        self.last_push = self.episode_time

    # ----- acting -----
    def _noise_scale(self):
        """Exploration noise of the next action, times the learned one: low while it should stand still."""
        standing = ((self.stage.name == "one_leg" and M.one_leg_still(self.motion_time))
                    or (self.stage.name == "stand" and min(self.episode_time, self.episode_time - self.last_push) > 1.0))
        noise = self.trainer.noise_std() if self.trainer is not None else 0.0
        return min(1.0, STILL_NOISE_STD / noise) if standing and noise > 0.0 else 1.0

    def _apply(self, reference, action):
        """Targets = reference motion + the policy's correction."""
        for servo, ref, a in zip(self.servos, reference, action):
            servo.command(ref + min(max(float(a), -1.0), 1.0) * RESIDUAL_SPAN_DEG)

    def Update(self, dt):
        if self.process_end_request():
            self.falls += 1
            self.stats.count("fall_touch")
            self._count_fall_phase()
            self._finish(terminated=True, survived=False)
            return
        if self._simulation_blew_up():
            # Not the policy's fault: end like a time limit (bootstrapped from the last good observation).
            self.stats.count("fall_sim-unstable")
            if self.has_active_action:
                self.end_episode(terminated=False, final_observation=self.last_observation)
            else:
                self.begin_episode()
            return
        self.clock += self.control_dt
        self.episode_time += self.control_dt
        if not self._waiting():
            self.motion_time += self.control_dt

        corners = self._foot_corners()
        contacts = np.array([1.0 if c[1] < 0.003 else 0.0 for c in corners], dtype=np.float32)
        if self.stage.name == "one_leg":
            self._note_lift_start()
        parts, metrics, (left_down, right_down) = self._reward(corners, contacts, self._reference(self.motion_time))
        waiting = self._waiting()
        if waiting:                                 # only the streak pays here (no alive bonus, see STEADY_POINT)
            reward, kept = self._steady_step(), 0.0  # (also moves the motion on: even without a policy acting)
            parts = {"steady": reward}
        else:
            reward, kept = combine_reward(parts)
        if self.has_active_action:                  # a reward before the first action has nobody to credit
            self.add_reward(reward)
            metrics["alive_kept"] = kept
            metrics["waiting"] = float(waiting)
            if not self.is_test:                    # the report describes training; tests go to the curriculum
                self.stats.record_support(left_down, right_down)
                self.stats.record(self.stage.name, parts, metrics)

        reason = self._fall_reason()
        if reason:
            self.add_reward(FALL_PENALTY)
            self.falls += 1
            self.stats.count("fall_" + reason)
            self._count_fall_phase()
            self._finish(terminated=True, survived=False)
            return

        # The command sent now is reached during the next control step, so aim at the reference there.
        # (while a pause waits, the motion there holds still: the same targets)
        reference = self._reference(self.motion_time + (0.0 if self._waiting() else self.control_dt))
        observation = self.sense(contacts, reference)
        if self.motion_time >= self.stage.seconds or self.episode_time >= MAX_EPISODE_SECONDS:
            self._finish(terminated=False, survived=True,
                         final_observation=mirror_observation(observation) if self._mirrored() else observation)
            return
        if self.pushes and self.stage.push_speed > 0 and self.episode_time >= self.next_push:
            self._push()

        mirrored = self._mirrored()
        self.noise_scale = self._noise_scale()
        self.add_observation(mirror_observation(observation) if mirrored else observation)
        action = np.array(self.get_continuous_actions(), dtype=np.float32, copy=True)
        if mirrored:
            action = np.asarray(mirror_action(action), dtype=np.float32)
        self._apply(reference, action)
        self.action_before_previous = self.previous_action
        self.previous_action = action
        self.trainer.learn_if_ready()
