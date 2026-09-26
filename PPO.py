"""
Multi-agent PPO for a pure-Python game engine.

Design:
- Trainer owns the shared ActorCritic model, rollout buffers, optimizer, and learning loop.
- Agent is the component you attach to each game object.
- Many Agent instances can share one Trainer, so they share one model and learn together.
- Agents may request actions independently; the trainer keeps separate trajectories per agent.
- Supports both continuous and discrete actions:
    - action_dim_continuous: number of bounded continuous action values.
    - action_dim_discrete: number of classes in one categorical action.

Expected engine flow per agent:
    agent.begin_episode()  # Start() may call this automatically
    while episode running:
        obs = np.ndarray shape [obs_dim]
        action = agent.get_actions(obs)
        # or:
        continuous_action, discrete_action = agent.get_mixed_actions(obs)
        # apply action to game object
        agent.add_reward(reward_delta)
        # after the reward has been added:
        # - call agent.end_episode() for a real terminal event;
        # - max_steps episodes end automatically when the next observation
        #   is passed to agent.get_actions().

Training flow:
    stats = trainer.learn_if_ready()

Call trainer.learn_if_ready() once per engine frame/tick, or after some number of agent actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Deque, Dict, List, Optional, Tuple, Union
from collections import deque
from numbers import Real

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import (
    AffineTransform,
    Categorical,
    Distribution,
    Normal,
    TanhTransform,
    TransformedDistribution,
)


from bereshit import Vector3, Quaternion, Component

# Range of the continuous actions' log std. The parameter itself is kept inside it (see Trainer._cap_noise):
# clamping it in forward() instead cut its gradient off, so a log_std past a bound stayed there for good.
LOG_STD_MIN = math.log(1e-4)
LOG_STD_MAX = math.log(2.0)

@dataclass
class Config:
    obs_dim: int
    action_dim_continuous: int = 0
    action_dim_discrete: int = 0
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 5
    minibatch_size: int = 256
    rollout_steps: int = 2048
    continuous_action_low: float = -1.0
    continuous_action_high: float = 1.0
    device: str = "cpu"
    performance_window: int = 100
    max_episode_reward: Optional[float] = None
    best_model_path: Optional[str] = None
    max_steps: Optional[int] = None
    # Normalize policy inputs with running mean/std (saved in the checkpoint, needed on the robot too).
    normalize_observations: bool = True
    observation_clip: float = 10.0
    # Stop an update early when the policy moves too far from the one that collected the data.
    target_kl: Optional[float] = 0.02
    # Initial exploration noise of continuous actions (log of the std before tanh). 0 -> std 1.
    initial_log_std: float = 0.0
    # Give the value estimate its own network. With a shared one, a large value error (returns
    # of tens of points) drags the policy along and the KL stop cuts most updates short.
    # Off by default so older checkpoints keep loading.
    separate_value_network: bool = False
    # Free-form facts about what the model was trained for (saved with it), e.g. {"servo": "STS3250_12V"}.
    metadata: Optional[Dict[str, object]] = None
    # Start with the continuous action mean at exactly 0 (last layer zeroed). For policies that output
    # corrections on top of a reference motion: an untrained policy then plays the reference itself.
    zero_initial_actions: bool = False
    # Symmetry loss weight (0 = off). With mirror functions set on the Trainer (set_mirror), the policy's
    # action for a mirrored observation is pulled toward the mirrored action, so a skill learned on one
    # side (e.g. standing on the right leg) also shapes the other side.
    symmetry_weight: float = 0.0
    # Upper limit on the continuous exploration noise (std before tanh; None = no limit), applied after
    # every optimizer step. Lower it over time to fine-tune a policy that already works: with a small
    # entropy bonus the learned std can still sit high for hours.
    max_noise_std: Optional[float] = None

    def __post_init__(self) -> None:
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if self.action_dim_continuous < 0:
            raise ValueError("action_dim_continuous must be >= 0")
        if self.action_dim_discrete < 0:
            raise ValueError("action_dim_discrete must be >= 0")
        if self.action_dim_continuous == 0 and self.action_dim_discrete == 0:
            raise ValueError("At least one of action_dim_continuous or action_dim_discrete must be > 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.continuous_action_low >= self.continuous_action_high:
            raise ValueError("continuous_action_low must be smaller than continuous_action_high")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be > 0")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be > 0")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0 when specified")
        if self.max_episode_reward is not None and self.max_episode_reward <= 0:
            raise ValueError("max_episode_reward must be > 0 when specified")
        if self.observation_clip <= 0:
            raise ValueError("observation_clip must be > 0")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 when specified")
        if self.max_noise_std is not None and self.max_noise_std <= 0:
            raise ValueError("max_noise_std must be > 0 when specified")


def latest_model_path(best_model_path: str) -> str:
    """Where the newest checkpoint is saved, next to the best one: model.pt -> modellatest.pt."""
    root, extension = os.path.splitext(best_model_path)
    return root + "latest" + extension


class RunningMeanStd:
    """Running mean/variance of observations (parallel Welford update)."""

    def __init__(self, size: int, epsilon: float = 1e-4):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, object]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ActorCritic(nn.Module):
    """Shared policy/value network for continuous, discrete, or mixed actions."""

    def __init__(
            self,
            obs_dim: int,
            action_dim_continuous: int,
            action_dim_discrete: int,
            hidden_size: int,
            continuous_action_low: float,
            continuous_action_high: float,
            initial_log_std: float = 0.0,
            separate_value_network: bool = False,
            zero_initial_actions: bool = False,
    ):
        super().__init__()
        self.action_dim_continuous = action_dim_continuous
        self.action_dim_discrete = action_dim_discrete
        self.continuous_action_low = float(continuous_action_low)
        self.continuous_action_high = float(continuous_action_high)

        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )

        self.backbone = mlp()
        # None: the critic reads the policy's backbone features.
        self.value_backbone = mlp() if separate_value_network else None

        self.critic = nn.Linear(hidden_size, 1)

        if action_dim_continuous > 0:
            self.actor_mean = nn.Linear(hidden_size, action_dim_continuous)
            initial_log_std = min(max(float(initial_log_std), LOG_STD_MIN), LOG_STD_MAX)
            self.log_std = nn.Parameter(torch.full((action_dim_continuous,), initial_log_std))
            if zero_initial_actions:
                nn.init.zeros_(self.actor_mean.weight)
                nn.init.zeros_(self.actor_mean.bias)
        else:
            self.actor_mean = None
            self.log_std = None

        if action_dim_discrete > 0:
            self.actor_logits = nn.Linear(hidden_size, action_dim_discrete)
        else:
            self.actor_logits = None

    def forward(self, obs: torch.Tensor, noise_scale: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """noise_scale [batch]: the exploration noise of each row times this (1 = the learned noise)."""
        x = self.backbone(obs)
        value_features = x if self.value_backbone is None else self.value_backbone(obs)
        out: Dict[str, torch.Tensor] = {"value": self.critic(value_features).squeeze(-1)}

        if self.action_dim_continuous > 0:
            mean = self.actor_mean(x)
            # log_std stays in [LOG_STD_MIN, LOG_STD_MAX]: the trainer clamps the parameter after every step.
            std = torch.exp(self.log_std).expand_as(mean)
            if noise_scale is not None:
                std = std * noise_scale.reshape(-1, 1)
            out["continuous_mean"] = mean
            out["continuous_std"] = std

        if self.action_dim_discrete > 0:
            out["discrete_logits"] = self.actor_logits(x)

        return out

    def distributions_and_value(self, obs: torch.Tensor, noise_scale: Optional[torch.Tensor] = None) -> Tuple[
        Optional[Distribution], Optional[Categorical], torch.Tensor]:
        out = self.forward(obs, noise_scale)

        continuous_dist = None
        discrete_dist = None

        if self.action_dim_continuous > 0:
            # The environment receives a bounded action. Model that bounded
            # distribution directly so PPO evaluates the action that was
            # actually applied, rather than an unclipped latent Normal sample.
            base_dist = Normal(out["continuous_mean"], out["continuous_std"])
            midpoint = (self.continuous_action_high + self.continuous_action_low) / 2.0
            half_range = (self.continuous_action_high - self.continuous_action_low) / 2.0
            continuous_dist = TransformedDistribution(
                base_dist,
                [
                    TanhTransform(cache_size=1),
                    AffineTransform(loc=midpoint, scale=half_range),
                ],
            )

        if self.action_dim_discrete > 0:
            discrete_dist = Categorical(logits=out["discrete_logits"])

        return continuous_dist, discrete_dist, out["value"]


class RolloutBuffer:
    """Stores transitions from all agents, then flattens them for PPO learning."""

    def __init__(self):
        self.agent_trajectories: Dict[Union[int, str], List[Dict[str, object]]] = {}
        self.total_steps: int = 0

    def add(self, agent_id: Union[int, str], transition: Dict[str, object]) -> None:
        if agent_id not in self.agent_trajectories:
            self.agent_trajectories[agent_id] = []
        self.agent_trajectories[agent_id].append(transition)
        self.total_steps += 1

    def __len__(self) -> int:
        return self.total_steps

    def clear(self) -> None:
        self.agent_trajectories.clear()
        self.total_steps = 0


class Trainer:
    """Owns the shared model and trains it from all agents' experiences."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCritic(
            obs_dim=config.obs_dim,
            action_dim_continuous=config.action_dim_continuous,
            action_dim_discrete=config.action_dim_discrete,
            hidden_size=config.hidden_size,
            continuous_action_low=config.continuous_action_low,
            continuous_action_high=config.continuous_action_high,
            initial_log_std=config.initial_log_std,
            separate_value_network=config.separate_value_network,
            zero_initial_actions=config.zero_initial_actions,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer()
        # Incremented by every learn(); lets store_transition() detect actions chosen by an older policy.
        self.policy_version: int = 0
        self.obs_rms = RunningMeanStd(config.obs_dim)
        self.update_obs_stats: bool = True
        self.mirror_observation = None   # callables on torch tensors, see set_mirror()
        self.mirror_action = None

        # Performance tracking.
        self.training_updates: int = 0
        self.total_environment_steps: int = 0
        self.completed_episodes: int = 0
        self.episode_rewards: Deque[float] = deque(maxlen=config.performance_window)
        self.episode_lengths: Deque[int] = deque(maxlen=config.performance_window)
        self.best_average_reward: float = float("-inf")
        self.last_stats: Dict[str, float] = {}

        # When True, the model is only used for actions.
        # No transitions are stored and learn_if_ready() will not train.
        self.inference_only: bool = False

    def set_mirror(self, mirror_observation, mirror_action) -> None:
        """Left/right mirror functions (torch tensors; observation raw, not normalized) for the symmetry loss."""
        self.mirror_observation = mirror_observation
        self.mirror_action = mirror_action

    def _mirror_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        """Mirror a batch of normalized observations: back to raw, mirror, normalize again."""
        if not self.config.normalize_observations:
            return self.mirror_observation(obs)
        mean = torch.as_tensor(self.obs_rms.mean, dtype=obs.dtype, device=obs.device)
        std = torch.as_tensor(np.sqrt(self.obs_rms.var + 1e-8), dtype=obs.dtype, device=obs.device)
        mirrored = self.mirror_observation(obs * std + mean)
        clip = self.config.observation_clip
        return ((mirrored - mean) / std).clamp(-clip, clip)

    def _to_obs_tensor(self, observation: np.ndarray) -> torch.Tensor:
        if not isinstance(observation, np.ndarray):
            observation = np.asarray(observation, dtype=np.float32)
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.config.obs_dim:
            raise ValueError(f"Expected observation size {self.config.obs_dim}, got shape {tuple(obs.shape)}")
        return obs

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """The observation as the network sees it (raw if normalization is off)."""
        observation = np.asarray(observation, dtype=np.float32)
        if not self.config.normalize_observations:
            return observation
        return self.obs_rms.normalize(observation, self.config.observation_clip)

    def _clamp_continuous_action(self, action: torch.Tensor) -> torch.Tensor:
        """Keep bounded actions away from +/-1 before evaluating tanh log-probs."""
        scale = max(
            1.0,
            abs(self.config.continuous_action_low),
            abs(self.config.continuous_action_high),
        )
        eps = min(
            float(torch.finfo(action.dtype).eps) * scale * 4.0,
            (self.config.continuous_action_high - self.config.continuous_action_low) * 0.25,
        )
        return action.clamp(
            self.config.continuous_action_low + eps,
            self.config.continuous_action_high - eps,
        )

    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or infinite values")

    def _cap_noise(self) -> None:
        """Keep log_std in [LOG_STD_MIN, LOG_STD_MAX] and the exploration noise at or below config.max_noise_std."""
        if self.model.log_std is None:
            return
        high = LOG_STD_MAX
        if self.config.max_noise_std is not None:
            high = min(high, math.log(self.config.max_noise_std))
        with torch.no_grad():
            self.model.log_std.clamp_(min=min(LOG_STD_MIN, high), max=high)

    def set_max_noise_std(self, value: Optional[float]) -> None:
        if value is not None and value <= 0:
            raise ValueError("max_noise_std must be > 0 when specified")
        self.config.max_noise_std = value
        self._cap_noise()

    def noise_std(self) -> float:
        """Mean exploration noise std of the continuous actions (0 without continuous actions)."""
        return 0.0 if self.model.log_std is None else float(self.model.log_std.detach().exp().mean())

    def _require_finite_model(self) -> None:
        for name, parameter in self.model.named_parameters():
            self._require_finite(f"model parameter {name}", parameter)

    @torch.no_grad()
    def act(
            self,
            observation: np.ndarray,
            deterministic: bool = False,
            noise_scale: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Dict[str, torch.Tensor]]:
        """
        Samples mixed actions from the shared policy.
        noise_scale: this action's exploration noise is the learned one times this (e.g. less noise where
        the task is standing still). Stored with the transition, so learning uses the same distribution.

        Returns:
            continuous_action:
                np.ndarray shape [action_dim_continuous], or None if no continuous branch.
            discrete_action:
                int in [0, action_dim_discrete - 1], or None if no discrete branch.
            ppo_data:
                tensors needed later for PPO storage.
        """
        if deterministic and not self.inference_only:
            raise ValueError("Use deterministic actions only in inference mode")

        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim != 1:
            raise ValueError(f"Expected a 1-D observation, got shape {observation.shape}")
        if not np.isfinite(observation).all():
            raise ValueError("Observation contains NaN or infinite values")

        # Statistics only follow the data while training; a trained model keeps the ones it learned with.
        # Parallel workers turn update_obs_stats off and let the learner update the shared statistics.
        if self.config.normalize_observations and not self.inference_only and self.update_obs_stats:
            self.obs_rms.update(observation)
        obs = self._to_obs_tensor(self.normalize_observation(observation))
        if not (np.isfinite(noise_scale) and noise_scale > 0.0):
            raise ValueError(f"noise_scale must be positive, got {noise_scale}")
        scale = torch.tensor([float(noise_scale)], device=self.device)
        continuous_dist, discrete_dist, value = self.model.distributions_and_value(obs, scale)

        log_prob_parts: List[torch.Tensor] = []
        entropy_parts: List[torch.Tensor] = []
        ppo_data: Dict[str, object] = {
            "value": value.squeeze(0),
            "policy_version": self.policy_version,
            "noise_scale": float(noise_scale),
            # Exactly what the network saw, so learning uses the same input the action was chosen from.
            "observation": obs.squeeze(0),
        }

        continuous_action_np: Optional[np.ndarray] = None
        discrete_action_int: Optional[int] = None

        if continuous_dist is not None:
            if deterministic:
                # TransformedDistribution.mean is not generally defined.
                # Transform the Normal mean explicitly for evaluation.
                base_mean = continuous_dist.base_dist.loc
                midpoint = (self.config.continuous_action_high + self.config.continuous_action_low) / 2.0
                half_range = (self.config.continuous_action_high - self.config.continuous_action_low) / 2.0
                continuous_action = torch.tanh(base_mean) * half_range + midpoint
            else:
                continuous_action = continuous_dist.sample()

            continuous_action = self._clamp_continuous_action(continuous_action)
            self._require_finite("sampled continuous action", continuous_action)
            continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
            # TransformedDistribution does not provide a closed-form entropy.
            # The base Normal entropy is a stable, common approximation for the
            # exploration bonus.
            continuous_entropy = continuous_dist.base_dist.entropy().sum(dim=-1)
            self._require_finite("continuous log probability", continuous_log_prob)

            continuous_action_np = continuous_action.squeeze(0).cpu().numpy()

            # Store the bounded action that the environment actually used.
            ppo_data["continuous_action"] = continuous_action.squeeze(0)
            log_prob_parts.append(continuous_log_prob)
            entropy_parts.append(continuous_entropy)

        if discrete_dist is not None:
            discrete_action = torch.argmax(discrete_dist.probs, dim=-1) if deterministic else discrete_dist.sample()
            discrete_log_prob = discrete_dist.log_prob(discrete_action)
            discrete_entropy = discrete_dist.entropy()

            discrete_action_int = int(discrete_action.squeeze(0).cpu().item())

            ppo_data["discrete_action"] = discrete_action.squeeze(0)
            log_prob_parts.append(discrete_log_prob)
            entropy_parts.append(discrete_entropy)

        ppo_data["log_prob"] = torch.stack([x.squeeze(0) for x in log_prob_parts]).sum()
        ppo_data["entropy"] = torch.stack([x.squeeze(0) for x in entropy_parts]).sum()

        return continuous_action_np, discrete_action_int, ppo_data

    def store_transition(
            self,
            agent_id: Union[int, str],
            observation: np.ndarray,
            ppo_data: Dict[str, torch.Tensor],
            reward: float,
            done: bool,
            next_observation: Optional[np.ndarray],
            truncated: bool = False,
    ) -> None:
        # `observation` is the raw one; store the normalized input that act() actually used.
        obs_tensor = ppo_data["observation"].detach().cpu()
        self._require_finite("transition observation", obs_tensor)

        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError(f"Reward must be finite, got {reward_value}")

        # If there is no final observation, PPO treats terminal next value as 0.
        next_obs_tensor = None
        if next_observation is not None:
            next_obs_tensor = self._to_obs_tensor(self.normalize_observation(next_observation)).squeeze(0).detach().cpu()
            self._require_finite("next observation", next_obs_tensor)

        if ppo_data.get("policy_version") == self.policy_version:
            value_tensor = ppo_data["value"].detach().cpu()
            log_prob_tensor = ppo_data["log_prob"].detach().cpu()
        else:
            # The action was chosen before the last update (it was pending while learn() ran).
            # Re-evaluate it with the current policy so the batch doesn't mix two policies.
            value_tensor, log_prob_tensor = self._current_value_and_log_prob(obs_tensor, ppo_data)
        self._require_finite("stored value estimate", value_tensor)
        self._require_finite("stored log probability", log_prob_tensor)

        transition: Dict[str, object] = {
            "observation": obs_tensor,
            "reward": reward_value,
            "done": bool(done),
            # Time-limit end of an episode: not terminal, but the next transition belongs to a new episode.
            "truncated": bool(truncated),
            "next_observation": next_obs_tensor,
            "value": value_tensor,
            "log_prob": log_prob_tensor,
            "noise_scale": float(ppo_data.get("noise_scale", 1.0)),
        }

        if self.config.action_dim_continuous > 0:
            continuous_action = ppo_data["continuous_action"].detach().cpu()
            self._require_finite("stored continuous action", continuous_action)
            transition["continuous_action"] = self._clamp_continuous_action(continuous_action)

        if self.config.action_dim_discrete > 0:
            transition["discrete_action"] = ppo_data["discrete_action"].detach().cpu()

        self.buffer.add(agent_id, transition)
        self.total_environment_steps += 1

    @torch.no_grad()
    def _current_value_and_log_prob(
            self,
            obs_tensor: torch.Tensor,
            ppo_data: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Value and log-probability of an already chosen action under the current policy."""
        continuous_dist, discrete_dist, value = self.model.distributions_and_value(
            obs_tensor.to(self.device).unsqueeze(0),
            torch.tensor([float(ppo_data.get("noise_scale", 1.0))], device=self.device),
        )
        log_prob = torch.zeros(1, device=self.device)
        if continuous_dist is not None:
            action = self._clamp_continuous_action(ppo_data["continuous_action"].to(self.device).unsqueeze(0))
            log_prob = log_prob + continuous_dist.log_prob(action).sum(dim=-1)
        if discrete_dist is not None:
            log_prob = log_prob + discrete_dist.log_prob(ppo_data["discrete_action"].to(self.device).unsqueeze(0))
        return value.squeeze(0).cpu(), log_prob.squeeze(0).cpu()

    @torch.no_grad()
    def _value_of_next_obs(self, next_observation: Optional[torch.Tensor], done: bool) -> torch.Tensor:
        if done or next_observation is None:
            return torch.tensor(0.0)
        next_observation = next_observation.to(self.device).unsqueeze(0)
        _, _, value = self.model.distributions_and_value(next_observation)
        return value.squeeze(0).detach().cpu()

    def _flatten_with_gae(self) -> Dict[str, torch.Tensor]:
        """
        Computes GAE separately for each agent trajectory, then flattens all agents.
        This is important because agents may request actions independently.
        """
        flat_obs: List[torch.Tensor] = []
        flat_log_probs: List[torch.Tensor] = []
        flat_values: List[torch.Tensor] = []
        flat_returns: List[torch.Tensor] = []
        flat_advantages: List[torch.Tensor] = []
        flat_continuous_actions: List[torch.Tensor] = []
        flat_discrete_actions: List[torch.Tensor] = []
        flat_noise_scales: List[float] = []

        for trajectory in self.buffer.agent_trajectories.values():
            if not trajectory:
                continue

            rewards = torch.tensor([t["reward"] for t in trajectory], dtype=torch.float32)
            values = torch.stack([t["value"] for t in trajectory]).float()

            advantages = torch.zeros_like(rewards)
            last_gae = torch.tensor(0.0)

            for i in reversed(range(len(trajectory))):
                transition = trajectory[i]
                done = bool(transition["done"])
                truncated = bool(transition["truncated"])

                if done:
                    next_value = torch.tensor(0.0)
                elif truncated or i == len(trajectory) - 1:
                    # Time limit or end of rollout: bootstrap from the observation after this action,
                    # not from values[i + 1], which after a reset is the first state of the next episode.
                    next_value = self._value_of_next_obs(transition["next_observation"], False)
                else:
                    next_value = values[i + 1]

                # An episode boundary (terminal or time limit) stops advantages leaking back from the next episode.
                episode_continues = 0.0 if (done or truncated) else 1.0
                delta = rewards[i] + self.config.gamma * next_value - values[i]
                last_gae = delta + self.config.gamma * self.config.gae_lambda * episode_continues * last_gae
                advantages[i] = last_gae

            returns = advantages + values

            for i, transition in enumerate(trajectory):
                flat_obs.append(transition["observation"])
                flat_log_probs.append(transition["log_prob"])
                flat_values.append(transition["value"])
                flat_advantages.append(advantages[i])
                flat_returns.append(returns[i])
                flat_noise_scales.append(float(transition.get("noise_scale", 1.0)))

                if self.config.action_dim_continuous > 0:
                    flat_continuous_actions.append(transition["continuous_action"])
                if self.config.action_dim_discrete > 0:
                    flat_discrete_actions.append(transition["discrete_action"])

        batch: Dict[str, torch.Tensor] = {
            "observations": torch.stack(flat_obs).float().to(self.device),
            "old_log_probs": torch.stack(flat_log_probs).float().to(self.device),
            "advantages": torch.stack(flat_advantages).float().to(self.device),
            "returns": torch.stack(flat_returns).float().to(self.device),
            "noise_scales": torch.tensor(flat_noise_scales, dtype=torch.float32, device=self.device),
        }

        if self.config.action_dim_continuous > 0:
            batch["continuous_actions"] = torch.stack(flat_continuous_actions).float().to(self.device)

        if self.config.action_dim_discrete > 0:
            batch["discrete_actions"] = torch.stack(flat_discrete_actions).long().to(self.device)

        return batch

    def learn_if_ready(self, force: bool = False) -> Optional[Dict[str, float]]:
        """Train once when enough shared rollout data exists."""
        if self.inference_only:
            return None
        if len(self.buffer) < self.config.rollout_steps and not force:
            return None
        if len(self.buffer) == 0:
            return None
        stats = self.learn()
        self.last_stats = stats
        print("PPO update:", stats)
        self.print_performance()
        return stats

    def record_episode_result(self, episode_reward: float, episode_length: int) -> None:
        """
        Called by Agent.end_episode().

        Tracks recent performance and saves the shared model whenever the average
        episode reward over the recent window is the best so far.
        """
        self.completed_episodes += 1
        self.episode_rewards.append(float(episode_reward))
        self.episode_lengths.append(int(episode_length))




    def get_average_reward(self) -> Optional[float]:
        if len(self.episode_rewards) == 0:
            return None
        return float(np.mean(self.episode_rewards))

    def get_average_episode_length(self) -> Optional[float]:
        if len(self.episode_lengths) == 0:
            return None
        return float(np.mean(self.episode_lengths))

    def print_performance(self) -> None:
        """
        Print a simple report showing how the shared PPO model is doing.

        You can call this every few seconds, every N frames, or after trainer.learn_if_ready().
        """
        avg_reward = self.get_average_reward()
        avg_length = self.get_average_episode_length()

        avg_reward_text = "N/A" if avg_reward is None else f"{avg_reward:.3f}"
        avg_length_text = "N/A" if avg_length is None else f"{avg_length:.1f}"
        best_reward_text = "N/A" if self.best_average_reward == float("-inf") else f"{self.best_average_reward:.3f}"

        print("=" * 48)
        print("PPO Performance")
        print(f"Updates:              {self.training_updates}")
        print(f"Environment steps:    {self.total_environment_steps}")
        print(f"Completed episodes:   {self.completed_episodes}")
        print(f"Avg reward/window:    {avg_reward_text}")
        print(f"Best avg reward:      {best_reward_text}")
        print(f"Avg episode length:   {avg_length_text}")
        print(f"Best model path:      {self.config.best_model_path}")
        print(f"Max steps/episode:    {self.config.max_steps if self.config.max_steps is not None else 'None'}")

        if self.last_stats:
            print(f"Policy loss:          {self.last_stats.get('policy_loss', 0.0):.5f}")
            print(f"Value loss:           {self.last_stats.get('value_loss', 0.0):.5f}")
            print(f"Entropy:              {self.last_stats.get('entropy', 0.0):.5f}")
            print(f"Last trained steps:   {self.last_stats.get('trained_steps', 0.0):.0f}")

        print("=" * 48)

    def learn(self) -> Dict[str, float]:
        cfg = self.config
        self._require_finite_model()
        batch = self._flatten_with_gae()

        observations = batch["observations"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        for name, tensor in (
                ("observations", observations),
                ("old log probabilities", old_log_probs),
                ("advantages", advantages),
                ("returns", returns),
        ):
            self._require_finite(name, tensor)

        advantage_std = advantages.std(unbiased=False).clamp_min(1e-8)
        advantages = (advantages - advantages.mean()) / advantage_std

        batch_size = observations.shape[0]
        indices = np.arange(batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0
        approx_kl = 0.0
        clip_fraction = 0.0
        epochs_run = 0
        stopped_early = False
        total_symmetry = 0.0

        for _ in range(cfg.update_epochs):
            if stopped_early:
                break
            epochs_run += 1
            np.random.shuffle(indices)

            for start in range(0, batch_size, cfg.minibatch_size):
                mb_idx = indices[start: start + cfg.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = observations[mb_idx_t]
                mb_old_log_probs = old_log_probs[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns[mb_idx_t]

                continuous_dist, discrete_dist, values = self.model.distributions_and_value(
                    mb_obs, batch["noise_scales"][mb_idx_t])

                new_log_prob_parts: List[torch.Tensor] = []
                entropy_parts: List[torch.Tensor] = []

                if continuous_dist is not None:
                    mb_continuous_actions = self._clamp_continuous_action(
                        batch["continuous_actions"][mb_idx_t]
                    )
                    continuous_log_probs = continuous_dist.log_prob(mb_continuous_actions).sum(dim=-1)
                    continuous_entropy = continuous_dist.base_dist.entropy().sum(dim=-1)
                    new_log_prob_parts.append(continuous_log_probs)
                    entropy_parts.append(continuous_entropy)

                if discrete_dist is not None:
                    mb_discrete_actions = batch["discrete_actions"][mb_idx_t]
                    discrete_log_probs = discrete_dist.log_prob(mb_discrete_actions)
                    discrete_entropy = discrete_dist.entropy()
                    new_log_prob_parts.append(discrete_log_probs)
                    entropy_parts.append(discrete_entropy)

                new_log_probs = torch.stack(new_log_prob_parts, dim=0).sum(dim=0)
                entropy = torch.stack(entropy_parts, dim=0).sum(dim=0).mean()

                # Avoid exp overflow before PPO's ratio clipping is applied.
                log_ratio = torch.clamp(new_log_probs - mb_old_log_probs, -20.0, 20.0)
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                    clip_fraction = float(((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean())
                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    # The policy already moved too far from the one that collected this data.
                    stopped_early = True
                    break

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = nn.functional.mse_loss(values, mb_returns)
                loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
                if cfg.symmetry_weight > 0 and self.mirror_observation is not None and self.action_dim_continuous_ok():
                    mean = self.model.forward(mb_obs)["continuous_mean"]
                    mirrored_mean = self.model.forward(self._mirror_normalized(mb_obs))["continuous_mean"]
                    symmetry_loss = nn.functional.mse_loss(mirrored_mean, self.mirror_action(mean))
                    loss = loss + cfg.symmetry_weight * symmetry_loss
                    total_symmetry += float(symmetry_loss.detach().cpu())

                self._require_finite("new log probabilities", new_log_probs)
                self._require_finite("PPO loss", loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.max_grad_norm,
                    error_if_nonfinite=False,
                )
                self._require_finite("gradient norm", gradient_norm)
                self.optimizer.step()
                self._cap_noise()
                self._require_finite_model()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy += float(entropy.detach().cpu())
                updates += 1

        self.buffer.clear()
        self.policy_version += 1
        average_reward = self.get_average_reward()
        # Only compare once the window holds enough episodes: right after a start or --resume a lucky
        # average over a handful of episodes could become a "best" that later models can't beat.
        enough = len(self.episode_rewards) >= min(20, self.config.performance_window)
        if enough and average_reward is not None and average_reward > self.best_average_reward:
            self.best_average_reward = average_reward
            if self.config.best_model_path is not None:
                self.save(self.config.best_model_path)

        if self.training_updates % 10 == 0 and self.config.best_model_path is not None:
            self.save(latest_model_path(self.config.best_model_path))
        self.training_updates += 1

        return {
            "policy_loss": total_policy_loss / max(updates, 1),
            "value_loss": total_value_loss / max(updates, 1),
            "entropy": total_entropy / max(updates, 1),
            "trained_steps": float(batch_size),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "epochs_run": float(epochs_run),
            "stopped_early": float(stopped_early),
            "symmetry_loss": total_symmetry / max(updates, 1),
        }

    def action_dim_continuous_ok(self) -> bool:
        return self.config.action_dim_continuous > 0

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "training_updates": self.training_updates,
                "total_environment_steps": self.total_environment_steps,
                "completed_episodes": self.completed_episodes,
                "best_average_reward": self.best_average_reward,
                "normalize_observations": self.config.normalize_observations,
                "obs_rms": self.obs_rms.state_dict(),
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True, load_stats: bool = True) -> None:
        """
        Load a saved model checkpoint.

        Use load_optimizer=True when continuing training.
        Use load_optimizer=False when only using the trained model for gameplay/inference.
        """
        checkpoint = torch.load(path, map_location=self.device)
        grown = self._grow_observations(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        self._cap_noise()   # a checkpoint from before log_std was kept in range

        if "obs_rms" in checkpoint:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])
            self.config.normalize_observations = bool(checkpoint.get("normalize_observations", True))
        elif self.config.normalize_observations:
            # Trained before normalization existed: the network expects raw observations.
            print(f"{path} has no observation statistics; using raw observations for this model.")
            self.config.normalize_observations = False

        if load_optimizer and "optimizer" in checkpoint and not grown:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # The saved state carries the old learning rate: the one in the config wins (it can be lowered
            # for fine-tuning without losing the optimizer's moment estimates).
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate
        if load_stats:
            self.training_updates = int(checkpoint.get("training_updates", self.training_updates))
            self.total_environment_steps = int(checkpoint.get("total_environment_steps", self.total_environment_steps))
            self.completed_episodes = int(checkpoint.get("completed_episodes", self.completed_episodes))
            self.best_average_reward = float(checkpoint.get("best_average_reward", self.best_average_reward))

    def _grow_observations(self, checkpoint: Dict[str, object]) -> bool:
        """
        A checkpoint trained with fewer observations, when new ones were appended at the end of the
        observation vector: the new inputs get zero weights (so the model acts exactly as before and learns
        to use them) and neutral normalization statistics. The optimizer state no longer fits and is dropped.
        Returns True if the checkpoint was grown.
        """
        state = checkpoint["model"]
        grown = False
        for key in ("backbone.0.weight", "value_backbone.0.weight"):
            weight = state.get(key)
            if weight is None or weight.shape[1] >= self.config.obs_dim:
                continue
            extra = self.config.obs_dim - weight.shape[1]
            state[key] = torch.cat([weight, torch.zeros(weight.shape[0], extra, dtype=weight.dtype,
                                                        device=weight.device)], dim=1)
            grown = True
        rms = checkpoint.get("obs_rms")
        if grown and rms is not None and len(rms["mean"]) < self.config.obs_dim:
            extra = self.config.obs_dim - len(rms["mean"])
            rms["mean"] = list(rms["mean"]) + [0.0] * extra
            rms["var"] = list(rms["var"]) + [1.0] * extra
        if grown:
            print(f"model trained with {self.config.obs_dim - extra} observations, now {self.config.obs_dim}: "
                  f"the new ones start with zero weight (optimizer state reset)")
        return grown

    def load_trained_model(self, path: Optional[str] = None, inference_only: bool = True) -> None:
        """
        Load a trained model and prepare it for use.

        By default this uses config.best_model_path and switches to inference-only mode.
        In inference-only mode:
        - actions still work
        - rewards are ignored for training
        - transitions are not stored
        - learn_if_ready() does nothing
        """
        model_path = path if path is not None else self.config.best_model_path
        if model_path is None:
            raise ValueError("No model path was provided and best_model_path is not configured")
        self.load(model_path, load_optimizer=False)
        self.set_inference_mode(inference_only)

    def set_inference_mode(self, enabled: bool = True) -> None:
        """
        Turn inference-only mode on/off.

        enabled=True:
            Use the trained model without learning.
        enabled=False:
            Resume normal PPO training behavior.
        """
        self.inference_only = bool(enabled)
        self.model.eval() if self.inference_only else self.model.train()
        if self.inference_only:
            self.buffer.clear()

    @torch.no_grad()
    def predict_actions(self, observation: np.ndarray, deterministic: bool = True) -> Dict[str, object]:
        """
        Use the current model directly without storing PPO data.

        This is the simplest function for gameplay after training.
        """
        previous_inference_mode = self.inference_only
        self.inference_only = True
        self.model.eval()
        try:
            continuous_action, discrete_action, _ = self.act(observation, deterministic=deterministic)
        finally:
            self.inference_only = previous_inference_mode
            self.model.eval() if previous_inference_mode else self.model.train()
        return {"continuous": continuous_action, "discrete": discrete_action}

class Academy:
    # Require explicit configuration. A silent default with obs_dim=1 and a
    # one-class action space can make an incorrectly configured walker appear
    # to train while it cannot produce meaningful actions.
    __ID = 0
    __trainer: Optional[Trainer] = None
    __Agents: List["Agent"] = []

    def __init__(self, trainer):
        self.Agents = []
        self.trainer = trainer

    @staticmethod
    def setup_trainer(config: Config) -> None:
        if Academy.__Agents:
            raise RuntimeError("Call Academy.setup_trainer() before creating Agent instances")
        Academy.__trainer = Trainer(config)

    @staticmethod
    def get_trainer() -> Trainer:
        if Academy.__trainer is None:
            raise RuntimeError("Call Academy.setup_trainer(config) before creating agents")
        return Academy.__trainer

    @staticmethod
    def load_model(model, load_optimizer=True, load_stats=True):
        """Load a checkpoint to continue training. To only run a trained model, use load_trained_model()."""
        Academy.get_trainer().load(model, load_optimizer, load_stats)

    @staticmethod
    def load_trained_model(model=None):
        """
        Load a checkpoint to run it: no training, no saving over the file,
        and agents take the policy's best (deterministic) action instead of sampling.
        """
        Academy.get_trainer().load_trained_model(model, inference_only=True)

    @staticmethod
    def AddAgent(agent: "Agent") -> None:
        if Academy.__trainer is None:
            raise RuntimeError("Call Academy.setup_trainer(config) before creating agents")
        agent.agent_id = Academy.__ID
        Academy.__ID += 1
        agent.trainer = Academy.__trainer
        Academy.__Agents.append(agent)

class Agent(Component):
    """
    Game-object component class.

    Attach one Agent to each game object.
    All agents may share the same Trainer to learn one shared policy.
    """
    def __init__(self):
        self.trainer = None
        self.agent_id = 0
        self._end_requested: bool = False
        super(Agent, self).__init__()
        Academy.AddAgent(self)
        self.episode_reward: float = 0.0
        self.pending_reward: float = 0.0
        self.episode_step: int = 0

        self.last_observation: Optional[np.ndarray] = None
        self.last_ppo_data: Optional[Dict[str, torch.Tensor]] = None
        self.has_active_action: bool = False
        # Exploration noise of this agent's next actions, times the learned one (see Trainer.act).
        self.noise_scale: float = 1.0

        self.observations = np.empty(self.trainer.config.obs_dim)
        self.collected_observations = 0

    def Start(self):
        self.begin_episode()

    def begin_episode(self) -> None:
        """Reset this agent and then invoke the user episode-start hook."""
        self.__OnEpisodeBegin()

    def get_collected_observations_length(self):
        return self.collected_observations

    def _append_observation_values(self, values) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        end = self.collected_observations + values.size
        if end > self.trainer.config.obs_dim:
            raise ValueError(
                f"Observation buffer overflow: adding {values.size} values would "
                f"exceed obs_dim={self.trainer.config.obs_dim}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Observation contains NaN or infinite values")
        self.observations[self.collected_observations:end] = values
        self.collected_observations = end

    def add_observation(self, observation):
        if isinstance(observation, Vector3):
            self._append_observation_values((observation.x, observation.y, observation.z))
        elif isinstance(observation, Quaternion):
            self._append_observation_values((observation.x, observation.y, observation.z, observation.w))
        elif isinstance(observation, Real):
            self._append_observation_values((observation,))
        elif isinstance(observation, (np.ndarray, list, tuple)):
            self._append_observation_values(observation)
        else:
            raise TypeError(f"observation type: {type(observation)} is not acceptable")

    def OnEpisodeBegin(self):
        pass

    def __OnEpisodeBegin(self) -> None:
        """
        Call this from the engine when this agent/game object starts or resets an episode.
        Unity-style capitalization is kept because you requested OnEpisodeBegin.
        """
        self.episode_reward = 0.0
        self.pending_reward = 0.0
        self.episode_step = 0
        self.collected_observations = 0
        self.last_observation = None
        self.last_ppo_data = None
        self.has_active_action = False
        self._end_requested = False
        self.OnEpisodeBegin()

    def get_continuous_actions(self, deterministic: Optional[bool] = None) -> np.ndarray:
        if self.collected_observations != self.trainer.config.obs_dim:
            collected_observations = self.collected_observations
            self.collected_observations = 0
            raise ValueError(f"Expected observation size {self.trainer.config.obs_dim}, got shape {collected_observations}")
        self.collected_observations = 0
        """
        Required function: returns only the continuous action branch.

        Use this if your agent has continuous actions only, or if the engine separately asks
        for continuous actions. If action_dim_continuous is 0, this raises an error.
        """
        continuous_action, _ = self.get_mixed_actions(self.observations, deterministic=deterministic)
        if continuous_action is None:
            raise RuntimeError("This Agent has no continuous action branch. Set action_dim_continuous > 0.")
        return continuous_action

    def get_discrete_action(self, observations: np.ndarray, deterministic: Optional[bool] = None) -> int:
        """
        Returns only the discrete action branch.

        Use this if your agent has discrete actions only, or if the engine separately asks
        for discrete actions. If action_dim_discrete is 0, this raises an error.
        """
        _, discrete_action = self.get_mixed_actions(observations, deterministic=deterministic)
        if discrete_action is None:
            raise RuntimeError("This Agent has no discrete action branch. Set action_dim_discrete > 0.")
        return discrete_action

    def get_actions(self, observations: np.ndarray, deterministic: Optional[bool] = None) -> Dict[str, object]:
        """
        Engine-friendly action getter.

        Returns:
            {
                "continuous": np.ndarray or None,
                "discrete": int or None,
            }
        """
        continuous, discrete = self.get_mixed_actions(observations, deterministic=deterministic)
        return {"continuous": continuous, "discrete": discrete}

    def get_mixed_actions(
            self,
            observations: np.ndarray,
            deterministic: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Gets both continuous and discrete action branches from the shared model.

        Important independent-agent pattern:
        - If this agent already had a previous action, this call stores that previous
          transition using the current observation as next_observation, unless the
          previous action reached max_steps and is closed as a truncation below.
        - Because this is per-agent state, agents can request actions independently.
        - Real terminal events are handled by the engine after the reward for the
          last action has been added.
        - A max_steps time-limit episode is ended automatically here, when the
          post-action observation arrives. This preserves the final reward and
          supplies the observation needed for value bootstrapping.
        - deterministic=None means: best action when running a trained model
          (inference mode), sampled action while training.
        """
        if deterministic is None:
            deterministic = self.trainer.inference_only
        # Copy: the caller may reuse its observation buffer, and this array is kept as last_observation.
        observations = np.array(observations, dtype=np.float32, copy=True)
        if observations.ndim != 1 or observations.shape[0] != self.trainer.config.obs_dim:
            raise ValueError(
                f"Expected observation shape ({self.trainer.config.obs_dim},), got {observations.shape}"
            )
        if not np.isfinite(observations).all():
            raise ValueError("Observation contains NaN or infinite values")

        # The previous action is complete now: its reward was added by the
        # engine and `observations` is the state after that action. If the
        # previous action reached max_steps, close the episode before asking
        # the policy for another action. This is automatic, but deliberately
        # happens one observation later so the final transition is not lost.
        if self.has_active_action and self.should_end_by_max_steps():
            self.end_episode(terminated=False, final_observation=observations)
            # end_episode() already reset the episode, so `observations` describe the old
            # episode. Don't act on them: hold still this step and act on the next fresh observation.
            return self._no_op_actions()

        if (
            not self.trainer.inference_only
            and self.has_active_action
            and self.last_observation is not None
            and self.last_ppo_data is not None
        ):
            self.trainer.store_transition(
                agent_id=self.agent_id,
                observation=self.last_observation,
                ppo_data=self.last_ppo_data,
                reward=self.pending_reward,
                done=False,
                next_observation=observations,
            )
            self.pending_reward = 0.0

        continuous_action, discrete_action, ppo_data = self.trainer.act(observations, deterministic=deterministic,
                                                                        noise_scale=self.noise_scale)

        self.last_observation = observations
        self.last_ppo_data = ppo_data
        self.has_active_action = True
        self.episode_step += 1

        return continuous_action, discrete_action

    def _no_op_actions(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Neutral actions (middle of the continuous range, discrete class 0) that are not recorded."""
        cfg = self.trainer.config
        continuous = None
        if cfg.action_dim_continuous > 0:
            midpoint = (cfg.continuous_action_high + cfg.continuous_action_low) / 2.0
            continuous = np.full(cfg.action_dim_continuous, midpoint, dtype=np.float32)
        discrete = 0 if cfg.action_dim_discrete > 0 else None
        return continuous, discrete

    def should_end_by_max_steps(self) -> bool:
        """Return True when this episode reached config.max_steps.

        This is exposed for diagnostics; get_mixed_actions() performs the
        automatic episode ending.
        """
        max_steps = self.trainer.config.max_steps
        return max_steps is not None and self.episode_step >= max_steps

    def add_reward(self, reward: float) -> None:
        """Add reward to the current action, optionally clipping the episode total."""
        if not self.has_active_action:
            # No action taken yet in this episode, so there is nothing to credit
            # (e.g. a leftover collision penalty right after a reset).
            return
        r = float(reward)
        limit = self.trainer.config.max_episode_reward
        if limit is not None:
            new_total = float(np.clip(self.episode_reward + r, -limit, limit))
            r = new_total - self.episode_reward
        self.pending_reward += r
        self.episode_reward += r

    def set_reward(self, reward: float) -> None:
        """Set this step's reward value, replacing any pending reward for the current step."""
        if not self.has_active_action:
            return   # nothing to credit yet, as in add_reward
        old_pending = self.pending_reward
        base_total = self.episode_reward - old_pending
        new_pending = float(reward)
        limit = self.trainer.config.max_episode_reward
        if limit is not None:
            new_total = float(np.clip(base_total + new_pending, -limit, limit))
            new_pending = new_total - base_total
        self.pending_reward = new_pending
        self.episode_reward = base_total + new_pending

    def request_end_episode(self, reward: float = 0.0) -> bool:
        """
        Ask to end the episode from inside the physics step (e.g. a collision callback).

        Resetting objects in the middle of the solver corrupts the step, so the
        episode actually ends on the next process_end_request() call from Update().
        Only the first request per episode counts; returns False for duplicates,
        e.g. when several body parts hit the floor in the same step.
        """
        if self._end_requested or not self.has_active_action:
            return False
        self._end_requested = True
        self.add_reward(reward)
        return True

    def process_end_request(self) -> bool:
        """Call at the start of Update(). Ends a requested episode; returns True if it did."""
        if not self._end_requested:
            return False
        self._end_requested = False
        self.end_episode()
        return True

    def end_episode(
            self,
            terminated: bool = True,
            final_observation: Optional[np.ndarray] = None,
    ) -> None:
        """
        End the current episode after its final reward has been added.

        terminated=True is a real terminal state and does not bootstrap.
        terminated=False represents a time-limit truncation and requires the
        observation after the final action so GAE can bootstrap correctly.
        """
        if not self.has_active_action:
            return
        if not terminated and final_observation is None:
            raise ValueError("A truncated episode requires final_observation")

        final_episode_reward = self.episode_reward
        final_episode_length = self.episode_step

        next_observation = None
        if final_observation is not None:
            next_observation = np.asarray(final_observation, dtype=np.float32)
            if next_observation.ndim != 1 or next_observation.shape[0] != self.trainer.config.obs_dim:
                raise ValueError(
                    f"Expected final observation shape ({self.trainer.config.obs_dim},), "
                    f"got {next_observation.shape}"
                )
            if not np.isfinite(next_observation).all():
                raise ValueError("Final observation contains NaN or infinite values")

        if (
            not self.trainer.inference_only
            and self.has_active_action
            and self.last_observation is not None
            and self.last_ppo_data is not None
        ):
            self.trainer.store_transition(
                agent_id=self.agent_id,
                observation=self.last_observation,
                ppo_data=self.last_ppo_data,
                reward=self.pending_reward,
                done=terminated,
                next_observation=next_observation,
                truncated=not terminated,
            )

        if not self.trainer.inference_only:
            self.trainer.record_episode_result(final_episode_reward, final_episode_length)
        # Store the terminal transition before allowing a rollout update.
        self.trainer.learn_if_ready()
        self.__OnEpisodeBegin()


# Example pure-Python engine setup:
if __name__ == "__main__":
    config = Config(
        obs_dim=12,
        action_dim_continuous=10,
        action_dim_discrete=0,
        rollout_steps=1024,
        max_steps=250,
        device="cpu",
    )
    Academy.setup_trainer(config)
    trainer = Academy.get_trainer()

    agents = [Agent() for _ in range(4)]

    for agent in agents:
        agent.begin_episode()

    for frame in range(5000):
        # Independent stepping example: not every agent needs to act every frame.
        for agent in agents:
            if np.random.random() < 0.75:
                obs = np.random.randn(config.obs_dim).astype(np.float32)
                actions = agent.get_actions(obs)

                continuous_action = actions["continuous"]
                discrete_action = actions["discrete"]

                # Your engine applies actions here.
                # Example:
                # game_object.apply_continuous_action(continuous_action)
                # game_object.apply_discrete_action(discrete_action)

                reward = np.random.randn() * 0.01
                agent.add_reward(reward)

                # A max_steps time-limit episode ends automatically on this
                # agent's next get_actions(obs) call, using that obs as the
                # post-action final observation.

            if np.random.random() < 0.005:
                agent.end_episode()

        stats = trainer.learn_if_ready()
        if stats is not None:
            print("PPO update:", stats)
            trainer.print_performance()

