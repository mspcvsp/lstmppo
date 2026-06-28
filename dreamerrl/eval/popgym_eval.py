import time
import numpy as np
import torch
from scipy.stats import entropy

# ============================================================
# 1. Deterministic Evaluation (your function, refined)
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
        obs = env.reset()["state"]
        world_state = world.init_state(batch_size).to(device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ep_return = torch.zeros(batch_size, device=device)

        while not torch.all(done):
            # RSSM update
            out = world.observe_step(world_state, obs)
            world_state = out["post"]

            # Greedy action
            logits = actor(world_state.h, world_state.z)
            action = torch.argmax(logits, dim=-1)

            # Step environment
            env_out = env.step(action)
            obs = env_out["state"]
            reward = env_out["reward"]
            done = env_out["is_terminal"]

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
    trainer.cfg.train.collect_steps = 1  # 1 env step per update
    trainer.cfg.train.random_exploration_steps = 0
    trainer.cfg.train.warmup_steps = 0

    wm_losses = []
    actor_losses = []
    critic_losses = []
    action_logits = []
    returns = []

    # Warm up replay buffer with enough steps to fill one episode
    for _ in range(200):
        trainer.collect_env_steps()

    start = time.time()
    print("\n")

    for step in range(steps):
        trainer.collect_env_steps()  # now exactly 1 step

        batch = trainer.replay.sample(trainer.cfg.train.batch_size)

        wm_losses.append(trainer.update_world_model(batch, step))
        a_loss, c_loss = trainer.update_actor_critic(batch, step)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        if trainer.env_state["is_last"].any():
            ep_return = trainer.env_state["reward"].sum().item()
            returns.append(ep_return)

        if step % 10 == 0 and step > 0:
            elapsed = time.time() - start
            eta = elapsed / step * (steps - step)
            print(f"[popgym] step={step}/{steps} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

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


def aggregate_popgym_results(results):
    wm_mean, wm_std, wm_cv = summarize([r["wm_loss"] for r in results])
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
