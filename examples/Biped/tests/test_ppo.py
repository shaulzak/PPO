"""
The PPO trainer: GAE with time limits, episode ends, stale actions, observation normalization, KL stop,
running a trained model, loading old checkpoints.
    python tests/test_ppo.py
"""
import math
import os
import tempfile

from _common import check, finish

import numpy as np
import torch
from bereshit.addons.PPO import Config, Academy, Agent, Trainer

print("test_ppo")

# 1. GAE: a time-limit end bootstraps from its final observation, not from the next episode
cfg = Config(obs_dim=2, action_dim_continuous=1, rollout_steps=10, gamma=0.9, gae_lambda=0.8,
             normalize_observations=False)
tr = Trainer(cfg)
V = {0: 1.0, 1: 2.0, 2: 3.0, 9: 7.0, 3: 100.0, 4: 200.0}
tr._value_of_next_obs = lambda nxt, done: torch.tensor(0.0) if (done or nxt is None) else torch.tensor(V[int(nxt[0])])


def add(o, r, done=False, trunc=False, nxt=None):
    tr.buffer.add(0, {"observation": torch.tensor([float(o), 0.0]), "reward": r, "done": done, "truncated": trunc,
                      "next_observation": None if nxt is None else torch.tensor([float(nxt), 0.0]),
                      "value": torch.tensor(V[o]), "log_prob": torch.tensor(0.0),
                      "continuous_action": torch.tensor([0.0])})


add(0, 1.0, nxt=1); add(1, 1.0, nxt=2); add(2, 1.0, trunc=True, nxt=9); add(3, 5.0, nxt=4); add(4, 5.0, nxt=9)
adv = tr._flatten_with_gae()["advantages"].tolist()
g, l = 0.9, 0.8
d2 = 1 + g * 7 - 3
d1 = 1 + g * 3 - 2
d0 = 1 + g * 2 - 1
expected = [d0 + g * l * (d1 + g * l * d2), d1 + g * l * d2, d2]
check("GAE bootstraps a truncated episode from its final observation",
      np.allclose(adv[:3], expected, atol=1e-4), f"{np.round(adv[:3], 4).tolist()} vs {np.round(expected, 4).tolist()}")

# 2-5. Agent behaviour (fresh Academy per process: run this file on its own)
cfg = Config(obs_dim=3, action_dim_continuous=2, rollout_steps=5, max_steps=3, minibatch_size=4, update_epochs=1)
Academy.setup_trainer(cfg)
tr = Academy.get_trainer()
a = Agent()
a.begin_episode()
buf = np.zeros(3, dtype=np.float32)
buf[:] = 1
a.get_mixed_actions(buf)
buf[:] = 99
check("stored observation is a copy of the caller's buffer", a.last_observation.tolist() == [1, 1, 1])
for v in (2, 3):
    buf[:] = v
    a.get_mixed_actions(buf)
buf[:] = 4
act, _ = a.get_mixed_actions(buf)
traj = tr.buffer.agent_trajectories[a.agent_id]
check("max_steps end: neutral unrecorded action, new episode starts clean",
      act.tolist() == [0.0, 0.0] and a.episode_step == 0 and traj[-1]["truncated"])
buf[:] = 5
a.get_mixed_actions(buf)
pending = a.last_ppo_data
tr.learn_if_ready(force=True)
buf[:] = 6
a.get_mixed_actions(buf)
stored = tr.buffer.agent_trajectories[a.agent_id][0]
_, lp_new = tr._current_value_and_log_prob(stored["observation"], pending)
check("an action chosen before an update is re-evaluated with the new policy",
      torch.allclose(stored["log_prob"], lp_new) and tr.policy_version == 1)

path = os.path.join(tempfile.gettempdir(), "biped_test_ppo.pt")
tr.save(path)
Academy.load_trained_model(path)
a.begin_episode()
o = np.array([0.1, 0.2, 0.3], dtype=np.float32)
x1, _ = a.get_mixed_actions(o)
x2, _ = a.get_mixed_actions(o)
check("a trained model runs deterministically, stores nothing and doesn't train",
      tr.inference_only and np.allclose(x1, x2) and len(tr.buffer) == 0 and tr.learn_if_ready(force=True) is None)

ck = torch.load(path)
ck.pop("obs_rms")
ck.pop("normalize_observations")
torch.save(ck, path)
tr.config.normalize_observations = True
Academy.load_trained_model(path)
check("a checkpoint from before normalization loads with raw observations", tr.config.normalize_observations is False)
os.remove(path)

# 6. zero_initial_actions: an untrained policy's best action is exactly 0 (it plays the reference motion)
t0 = Trainer(Config(obs_dim=5, action_dim_continuous=3, zero_initial_actions=True))
t0.inference_only = True
x, _, _ = t0.act(np.random.default_rng(1).normal(size=5).astype(np.float32), deterministic=True)
check("zero_initial_actions: untrained best action is 0", np.allclose(x, 0.0, atol=1e-5), f"{np.round(x, 6).tolist()}")

# 7. normalization statistics and the KL stop
cfg = Config(obs_dim=4, action_dim_continuous=2, rollout_steps=512, minibatch_size=64, target_kl=1e-5)
t2 = Trainer(cfg)
rng = np.random.default_rng(0)
for _ in range(600):
    t2.act(np.array([14, 0.5, -3, 100], dtype=np.float32) + rng.normal(0, [1, 0.1, 2, 30]).astype(np.float32))
check("running mean/std follow the data", np.allclose(t2.obs_rms.mean, [14, 0.5, -3, 100], atol=[0.2, 0.02, 0.3, 4]),
      f"{np.round(t2.obs_rms.mean, 2).tolist()}")

# 8. new observations appended at the end: an old checkpoint loads into the bigger network, acts exactly as
#    before whatever the new inputs say, and can go on training
small = Trainer(Config(obs_dim=4, action_dim_continuous=2, separate_value_network=True))
small.obs_rms.update(np.random.default_rng(2).normal(3.0, 2.0, size=(50, 4)))
path = os.path.join(tempfile.mkdtemp(), "small.pt")
small.save(path)
big = Trainer(Config(obs_dim=6, action_dim_continuous=2, separate_value_network=True))
big.load(path)
os.remove(path)
small.inference_only = big.inference_only = True
o = np.random.default_rng(3).normal(size=4).astype(np.float32)
a_small, _, i_small = small.act(o, deterministic=True)
a_big, _, i_big = big.act(np.concatenate([o, [0.7, -2.0]]).astype(np.float32), deterministic=True)
check("grown checkpoint: same action as before, the new inputs ignored at first", np.allclose(a_small, a_big, atol=1e-5),
      f"{np.round(a_small, 5).tolist()} vs {np.round(a_big, 5).tolist()}")
check("grown checkpoint: same value", abs(float(i_small["value"]) - float(i_big["value"])) < 1e-5)
check("grown checkpoint: new inputs get neutral statistics",
      np.allclose(big.obs_rms.mean[4:], 0.0) and np.allclose(big.obs_rms.var[4:], 1.0))

# 9. resuming keeps the optimizer state but takes the learning rate from the config (so it can be lowered)
fast = Trainer(Config(obs_dim=4, action_dim_continuous=2, learning_rate=3e-4))
path = os.path.join(tempfile.mkdtemp(), "fast.pt")
fast.save(path)
slow = Trainer(Config(obs_dim=4, action_dim_continuous=2, learning_rate=1e-4))
slow.load(path)
os.remove(path)
check("resumed training uses the config's learning rate, not the saved one",
      all(abs(g["lr"] - 1e-4) < 1e-12 for g in slow.optimizer.param_groups))


# 10. the exploration noise can be capped (fine-tuning): the cap holds right away and through updates
capped = Trainer(Config(obs_dim=3, action_dim_continuous=2, rollout_steps=64, minibatch_size=32,
                        initial_log_std=-1.0, entropy_coef=1.0))   # a big entropy bonus pushes the noise up
capped.set_max_noise_std(0.2)
check("noise cap applies at once", capped.noise_std() <= 0.2 + 1e-6, f"{capped.noise_std():.3f}")
rng = np.random.default_rng(4)
for _ in range(3):
    for _ in range(64):
        _, _, info = capped.act(rng.normal(size=3).astype(np.float32))
        capped.buffer.add(0, {"observation": info["observation"], "reward": 1.0, "done": False, "truncated": False,
                              "next_observation": None, "value": info["value"], "log_prob": info["log_prob"],
                              "continuous_action": info["continuous_action"]})
    capped.learn_if_ready(force=True)
check("noise cap holds through updates", capped.noise_std() <= 0.2 + 1e-6, f"{capped.noise_std():.3f}")

# 11. less noise for some actions (standing still): the spread follows noise_scale, and learning evaluates each
#     stored action with the noise it was chosen with (the first PPO ratio is exactly 1)
quiet = Trainer(Config(obs_dim=3, action_dim_continuous=2, initial_log_std=math.log(0.3), rollout_steps=64,
                       minibatch_size=64, update_epochs=1, normalize_observations=False))
o = np.array([0.2, -0.1, 0.4], dtype=np.float32)
full = np.array([quiet.act(o)[0] for _ in range(400)])
low = np.array([quiet.act(o, noise_scale=0.1)[0] for _ in range(400)])
spread = lambda a: np.arctanh(np.clip(a, -0.999999, 0.999999)).std(0).mean()    # before the tanh squashing
check("noise_scale 0.1: a tenth of the spread", 0.08 < spread(low) / spread(full) < 0.12, f"{spread(low) / spread(full):.3f}")
for k in range(64):
    _, _, info = quiet.act(o, noise_scale=0.1 if k % 2 else 1.0)
    quiet.store_transition(0, o, info, 1.0, False, o)
stored = quiet.buffer.agent_trajectories[0]
check("the noise scale is stored with each transition", [t["noise_scale"] for t in stored[:2]] == [1.0, 0.1])
batch = quiet._flatten_with_gae()
dist, _, _ = quiet.model.distributions_and_value(batch["observations"], batch["noise_scales"])
new = dist.log_prob(quiet._clamp_continuous_action(batch["continuous_actions"])).sum(-1)
check("learning re-evaluates each action with its own noise (ratio 1 before any update)",
      torch.allclose(new, batch["old_log_probs"], atol=1e-4), f"{float((new - batch['old_log_probs']).abs().max().detach()):.2e}")
stats = quiet.learn()
check("an update with mixed noise runs and stays finite", all(math.isfinite(v) for v in stats.values()))

finish()
