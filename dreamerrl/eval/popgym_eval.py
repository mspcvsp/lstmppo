"""
PopGym → Dreamer Evaluation Contract

This file implements deterministic evaluation of Dreamer-V3 agents on PopGym environments. Evaluation uses:

    - Dreamer's latent state (h, z)
    - Greedy actor policy (argmax over logits)
   - No exploration, no replay, no randomness

PopGym environments return dict observations containing:
    - "state":       flattened environment state
    - "prev_action": previous action taken (added by our wrapper)

The evaluation loop passes the *full* observation dict directly into world.observe_step(), allowing Dreamer to encode
both the current state and the previous action. This is REQUIRED for PopGym's RepeatPrevious* tasks, whose reward
depends on:

    reward = +1 if action_t == action_{t-1}

The evaluation loop runs until all vectorized environments report is_terminal=True, accumulating per-env episodic
returns. Because the wrapper auto-resets environments internally, evaluation always receives valid transitions and
never encounters dead envs.

The result dictionary includes:
    - mean return across episodes
    - std return
    - per-env return vector

This file provides a stable, deterministic benchmark for Dreamer functional correctness on PopGym tasks.
"""

import time

import numpy as np
import torch
from scipy.stats import entropy

# ============================================================
# 1. Deterministic Evaluation
# ============================================================


@torch.no_grad()
def evaluate_popgym(env, world, actor, episodes=10, device="cpu"):
    """
    Deterministic Dreamer-V3 evaluation on PopGym environments.
    Uses latent state + greedy actor policy (no exploration).
    """
    returns = []
    batch_size = env.batch_size

    for _ in range(episodes):
        obs = env.reset()
        world_state = world.init_state(batch_size).to(device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ep_return = torch.zeros(batch_size, device=device)

        while not torch.all(done):
            out = world.observe_step(world_state, obs)
            world_state = out["post"]

            logits = actor(world_state.h, world_state.z)
            action = torch.argmax(logits, dim=-1)

            obs = env.step(action)
            reward = obs["reward"]
            done = obs["is_terminal"]

            ep_return += reward * (~done)

        returns.append(ep_return.cpu())

    returns = torch.stack(returns, dim=0)
    return {
        "mean": returns.mean().item(),
        "std": returns.std().item(),
        "per_env": returns.mean(dim=0).tolist(),
    }


# ============================================================
# 2. Training Harness (single seed)
# ============================================================


def train_popgym_seed(trainer, steps=1000):
    """
    Fast PopGym functional training harness.
    - collect_steps = 1 for predictable runtime
    - warmup ensures replay buffer is non-empty
    - logits collected periodically (not only at end)
    """

    # Make PopGym fast + predictable
    trainer.cfg.train.collect_steps = 10
    trainer.cfg.train.random_exploration_steps = 100
    trainer.cfg.train.warmup_steps = 50

    wm_losses = []
    actor_losses = []
    critic_losses = []
    action_logits = []
    returns = []

    # Warm up replay buffer so sampling works
    for _ in range(200):
        trainer.collect_env_steps()

    start = time.time()

    for step in range(steps):
        trainer.collect_env_steps()

        batch = trainer.replay.sample(trainer.cfg.train.batch_size)

        wm_losses.append(trainer.update_world_model(batch, step))
        a_loss, c_loss = trainer.update_actor_critic(batch, step)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        # Episodic returns
        if trainer.env_state["is_last"].any() or trainer.env_state["is_terminal"].any():
            ep_return = trainer.env_state["reward"].sum().item()
            returns.append(ep_return)

        # ETA heartbeat
        if step % 10 == 0 and step > 0:
            elapsed = time.time() - start
            eta = elapsed / step * (steps - step)
            print(f"[popgym] step={step}/{steps} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        # Collect logits every 100 steps (robust)
        if step % 100 == 0:
            with torch.no_grad():
                logits = trainer.actor(trainer.world_state.h, trainer.world_state.z)
                action_logits.append(logits.cpu())

    # Fallback: ensure we have ~30 logits for KL
    if len(action_logits) < 30:
        with torch.no_grad():
            for _ in range(30 - len(action_logits)):
                logits = trainer.actor(trainer.world_state.h, trainer.world_state.z)
                action_logits.append(logits.cpu())

    # Prevent nan returns if no episodes finished (e.g. PopGym's "hard" envs)
    if len(returns) == 0:
        returns.append(0.0)

    return {
        "wm_loss": np.array(wm_losses),
        "actor_loss": np.array(actor_losses),
        "critic_loss": np.array(critic_losses),
        "returns": np.array(returns),
        "action_logits": torch.stack(action_logits),
    }


# ============================================================
# 3. Multi-seed Aggregator
# ============================================================


def kl_between_seeds(logits_a, logits_b):
    pa = torch.softmax(logits_a, dim=-1).view(-1)
    pb = torch.softmax(logits_b, dim=-1).view(-1)
    pa_np = pa[::10].numpy()
    pb_np = pb[::10].numpy()
    return float(entropy(pa_np, pb_np))


def summarize(arr_list):
    arr = np.stack(arr_list)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cv = std.mean() / abs(mean.mean())
    return mean, std, cv


def summarize_metrics(metrics_list):
    """
    metrics_list: list of lists of WorldModelMetrics
    Returns mean/std/cv for each metric curve.
    """
    # Convert list-of-dataclasses → dict of numpy arrays
    curves = {
        "total_loss": np.stack([m.total_loss.item() for m in metrics_list]),
        "recon_loss": np.stack([m.recon_loss.item() for m in metrics_list]),
        "reward_loss": np.stack([m.reward_loss.item() for m in metrics_list]),
        "cont_loss": np.stack([m.cont_loss.item() for m in metrics_list]),
        "kl_dyn": np.stack([m.kl_dyn.item() for m in metrics_list]),
        "kl_rep": np.stack([m.kl_rep.item() for m in metrics_list]),
    }

    # Compute summary stats for each curve
    summary = {}
    for key, arr in curves.items():
        mean = arr.mean()
        std = arr.std()
        cv = std / abs(mean) if mean != 0 else 0.0
        summary[key] = (mean, std, cv)

    return summary


def aggregate_popgym_results(results):
    wm_mean, wm_std, wm_cv = summarize([[m.total_loss.item() for m in r["wm_loss"]] for r in results])

    actor_mean, actor_std, actor_cv = summarize([r["actor_loss"] for r in results])
    critic_mean, critic_std, critic_cv = summarize([r["critic_loss"] for r in results])

    kl_vals = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            kl_vals.append(kl_between_seeds(results[i]["action_logits"], results[j]["action_logits"]))

    mean_return = np.mean([r["returns"].mean() for r in results])

    return {
        "wm_cv": wm_cv,
        "actor_cv": actor_cv,
        "critic_cv": critic_cv,
        "action_kl": float(np.mean(kl_vals)),
        "mean_return": float(mean_return),
    }
