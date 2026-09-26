"""
PPO trainer edge cases: exploration noise at its limit, the latest-checkpoint name, a reward before the first
action, an invalid noise cap.
    python tests/test_ppo_edges.py
"""
import os
import tempfile

from _common import check, finish

import numpy as np
import torch
from bereshit.addons.PPO import Config, Academy, Agent, Trainer

print("test_ppo_edges")

# 1. a noise std above the forward pass's limit (2) still learns and is reported as the one used. forward()
#    clamped log_std, so a log_std past the limit got no gradient and stayed there, and noise_std() reported
#    the unclamped value.
tr = Trainer(Config(obs_dim=2, action_dim_continuous=1, initial_log_std=1.0))
out = tr.model(torch.zeros(1, 2))
used = float(out["continuous_std"].detach().mean())
check("noise_std() is the std the policy samples with", abs(tr.noise_std() - used) < 1e-6,
      f"noise_std() {tr.noise_std():.3f}, sampled with {used:.3f}")
out["continuous_std"].sum().backward()
grad = tr.model.log_std.grad
check("log_std gets a gradient at the noise limit", grad is not None and float(grad.abs().sum()) > 0,
      f"gradient {None if grad is None else grad.tolist()}")

# 2. the newest checkpoint sits next to the best one for any extension: best_model_path[:-3] turned
#    "walker.pth" into "walker.platest.pt"
folder = tempfile.mkdtemp()
best = os.path.join(folder, "walker.pth")
cfg = Config(obs_dim=3, action_dim_continuous=2, rollout_steps=4, minibatch_size=4, update_epochs=1,
             best_model_path=best)
Academy.setup_trainer(cfg)
tr = Academy.get_trainer()
agent = Agent()
agent.begin_episode()
for i in range(6):
    agent.get_mixed_actions(np.full(3, 0.1 * i, dtype=np.float32))
    agent.add_reward(1.0)
tr.learn_if_ready(force=True)
saved = sorted(os.listdir(folder))
check("the latest checkpoint of walker.pth is walkerlatest.pth", saved == ["walkerlatest.pth"], f"saved {saved}")

# 3. a reward before the episode's first action is ignored, like add_reward: set_reward credited it to the
#    next episode's first transition
agent.begin_episode()
agent.set_reward(-5.0)
check("set_reward before the first action is ignored",
      agent.episode_reward == 0.0 and agent.pending_reward == 0.0,
      f"episode reward {agent.episode_reward}, pending {agent.pending_reward}")

# 4. a noise cap of 0 or less is refused up front: it raised math domain error inside learn(), mid-update
try:
    Config(obs_dim=2, action_dim_continuous=1, max_noise_std=0.0)
    refused = False
except ValueError:
    refused = True
check("Config refuses max_noise_std <= 0", refused)
before = tr.config.max_noise_std
try:
    tr.set_max_noise_std(0.0)
    refused, error = False, None
except ValueError as e:
    refused, error = True, e
check("set_max_noise_std(0) is refused and leaves the cap as it was",
      refused and tr.config.max_noise_std == before and "max_noise_std" in str(error),
      f"cap {tr.config.max_noise_std} (was {before}), error {error!r}")

finish()
