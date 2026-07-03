import time

import numpy as np
import pytest
import torch
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
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
            # Fallback: memory percent
            mem = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return 100 * mem / total
    return 0.0


def run_training(seed, steps, progress, task_id):
    cfg = make_default_config()
    cfg.train.seed = seed
    cfg.log.enable_wandb = False
    cfg.train.cuda = True
    cfg.train.deterministic_imagination = True
    cfg.train.deterministic_env = True
    cfg.env.num_envs = 4
    cfg.env.env_id = "popgym-RepeatFirstEasy-v0"

    # Dreamer-V3 requires fixed-length sequences in replay.
    # RepeatFirstEasy produces short episodes (~10 steps), so default seq_len=50 is invalid.
    # Setting seq_len = max_episode_steps = collect_steps ensures uniform slices and
    # stable replay buffer behavior, which is required for reproducibility tests.
    cfg.train.collect_steps = 10
    cfg.env.max_episode_steps = cfg.train.collect_steps
    cfg.train.seq_len = cfg.train.collect_steps

    cfg.env.deterministic = True
    cfg.env.seed = seed
    cfg.world.num_aux_reward_heads = 0

    set_global_seeds(seed)
    trainer = DreamerTrainer(cfg)

    returns = []
    wm_losses = []
    actor_losses = []
    critic_losses = []
    action_logits = []

    # Timing accumulators
    t_env = 0.0
    t_replay = 0.0
    t_world = 0.0
    t_actor = 0.0
    t_critic = 0.0

    for step in range(steps):
        progress.update(task_id, advance=1)

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

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
        transient=False,
        refresh_per_second=10,
    ) as progress:
        seed_tasks = {
            seed: progress.add_task(f"Seed {i + 1}/{len(seeds)}", total=steps) for i, seed in enumerate(seeds)
        }

        # Live dashboard
        table = Table(title="Dreamer‑V3 Reproducibility Dashboard")
        table.add_column("Metric")
        table.add_column("Value")

        with Live(table, refresh_per_second=4) as live:
            for seed in seeds:
                res = run_training(seed, steps, progress, seed_tasks[seed])
                progress.stop_task(seed_tasks[seed])
                results.append(res)

                # Update dashboard
                util = gpu_utilization()
                t = res["timing"]

                new_table = Table(title="Dreamer‑V3 Reproducibility Dashboard")
                new_table.add_column("Metric")
                new_table.add_column("Value")

                new_table.add_row("GPU Utilization (%)", f"{util:5.1f}")
                new_table.add_row("Env Time (s)", f"{t['env']:.3f}")
                new_table.add_row("Replay Time (s)", f"{t['replay']:.3f}")
                new_table.add_row("World Model Time (s)", f"{t['world']:.3f}")
                new_table.add_row("Actor Time (s)", f"{t['actor']:.3f}")
                new_table.add_row("Critic Time (s)", f"{t['critic']:.3f}")

                live.update(new_table)

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

    print("World Model CV:", wm_cv)
    print("Actor CV:", actor_cv)
    print("Critic CV:", critic_cv)

    kl_vals = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            kl_vals.append(kl_between_seeds(results[i]["action_logits"], results[j]["action_logits"]))

    print("Action KL mean:", np.mean(kl_vals))

    # CV = coefficient of variability (standard deviation / mean) across seeds for each metric

    # World model must be nearly deterministic
    wm_ok = wm_cv < 5e-3

    # Critic should have moderate variability
    critic_ok = 0.05 < critic_cv < 1.0

    # Actor is intentionally stochastic
    actor_ok = 0.5 < actor_cv < 3.0

    # Action KL should be healthy, not tiny
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
