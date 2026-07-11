import time
import numpy as np
import pytest
import torch
from scipy.stats import entropy

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import make_default_config


def gpu_utilization():
    """Return GPU utilization % if available, else memory usage."""
    if torch.cuda.is_available():
        try:
            util = torch.cuda.utilization()
            return util
        except Exception:
            mem = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return 100 * mem / total
    return 0.0


def run_training(seed, steps):
    cfg = make_default_config()

    cfg.train.seed = seed
    cfg.train.deterministic_imagination = True
    cfg.train.enforce_length_invariants = True
    cfg.train.deterministic_env = True
    cfg.train.cuda = True
    cfg.train.enable_repro_log = True  # logging OK now

    cfg.log.enable_wandb = False

    cfg.env.num_envs = 4
    cfg.env.env_id = "popgym-RepeatFirstEasy-v0"
    cfg.env.deterministic = True
    cfg.env.seed = seed
    cfg.env.parallel = False

    cfg.train.collect_steps = 10
    cfg.env.max_episode_steps = cfg.train.collect_steps
    cfg.train.seq_len = cfg.train.collect_steps

    cfg.world.num_aux_reward_heads = 0

    set_global_seeds(seed)
    trainer = DreamerTrainer(cfg)

    returns = []
    wm_losses = []
    actor_losses = []
    critic_losses = []
    action_logits = []

    t_env = 0.0
    t_replay = 0.0
    t_world = 0.0
    t_actor = 0.0
    t_critic = 0.0

    for step in range(steps):
        # ENV
        t0 = time.time()
        trainer.collect_env_steps()
        t_env += time.time() - t0

        # REPLAY
        t0 = time.time()
        batch = trainer.replay.sample(cfg.train.batch_size)
        t_replay += time.time() - t0

        # WORLD MODEL
        t0 = time.time()
        wm_metrics = trainer.update_world_model(batch, step)
        wm_losses.append(wm_metrics.total_loss.item())
        t_world += time.time() - t0

        # ACTOR + CRITIC
        t0 = time.time()
        a_loss, c_loss = trainer.update_actor_critic(batch, step)
        t_actor += time.time() - t0
        actor_losses.append(float(a_loss))

        t0 = time.time()
        critic_losses.append(float(c_loss))
        t_critic += time.time() - t0

        # Logits (last 50 steps only)
        if step >= steps - 50:
            with torch.no_grad():
                logits = trainer.actor(trainer.world_state.h, trainer.world_state.z)
                action_logits.append(logits)

        if trainer.env_state["is_last"].any():
            returns.append(trainer.env_state["reward"].sum().item())

    return {
        "returns": np.array(returns),
        "wm_loss": np.array(wm_losses),
        "actor_loss": np.array(actor_losses),
        "critic_loss": np.array(critic_losses),
        "action_logits": torch.stack(action_logits),
        "timing": {
            "env": t_env,
            "replay": t_replay,
            "world": t_world,
            "actor": t_actor,
            "critic": t_critic,
        },
    }


def run_all_seeds(seeds, steps):
    results = []
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        res = run_training(seed, steps)
        results.append(res)

        util = gpu_utilization()
        t = res["timing"]

        print(f"GPU Utilization: {util:5.1f}%")
        print(f"Env Time:    {t['env']:.3f}s")
        print(f"Replay Time: {t['replay']:.3f}s")
        print(f"World Time:  {t['world']:.3f}s")
        print(f"Actor Time:  {t['actor']:.3f}s")
        print(f"Critic Time: {t['critic']:.3f}s")

    return results


def kl_between_seeds(logits_a, logits_b):
    pa = torch.softmax(logits_a, dim=-1)
    pb = torch.softmax(logits_b, dim=-1)

    pa_np = pa.view(-1)[::10].detach().cpu().numpy()
    pb_np = pb.view(-1)[::10].detach().cpu().numpy()

    return float(entropy(pa_np, pb_np))


def summarize(metric_list):
    arr = np.stack(metric_list)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cv = std.mean() / abs(mean.mean())
    return mean, std, cv


@pytest.mark.manual
@pytest.mark.reproducibility
def test_reproducibility():
    seeds = [0, 1, 2]
    results = run_all_seeds(seeds, steps=300)

    wm_mean, wm_std, wm_cv = summarize([r["wm_loss"] for r in results])
    actor_mean, actor_std, actor_cv = summarize([r["actor_loss"] for r in results])
    critic_mean, critic_std, critic_cv = summarize([r["critic_loss"] for r in results])

    print("\nWorld Model CV:", wm_cv)
    print("Actor CV:", actor_cv)
    print("Critic CV:", critic_cv)

    kl_vals = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            kl_vals.append(kl_between_seeds(results[i]["action_logits"], results[j]["action_logits"]))

    print("Action KL mean:", np.mean(kl_vals))

    wm_ok = wm_cv < 5e-3
    critic_ok = 0.05 < critic_cv < 1.0
    actor_ok = 0.5 < actor_cv < 3.0

    mean_kl = np.mean(kl_vals)
    cfg = make_default_config()

    if cfg.world.num_aux_reward_heads > 0:
        kl_ok = mean_kl < 6.0
    else:
        kl_ok = 0.1 < mean_kl < 2.0

    if wm_ok and critic_ok and actor_ok and kl_ok:
        print("\n✅ Statistical reproducibility PASSED.")
    else:
        print("\n❌ Statistical reproducibility FAILED.")
