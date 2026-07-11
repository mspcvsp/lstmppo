import numpy as np
import pytest
import torch
from loguru import logger
from scipy.stats import entropy

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import make_default_config

logger.remove()  # Remove ALL sinks globally before anything else


def run_training(seed, steps):
    cfg = make_default_config()

    cfg.train.seed = seed
    cfg.train.deterministic_imagination = True
    cfg.train.enforce_length_invariants = True
    cfg.train.deterministic_env = True
    cfg.train.cuda = True

    # Disable reproducibility logging for this statistical test
    cfg.train.enable_repro_log = False

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

    for step in range(steps):
        trainer.collect_env_steps()

        batch = trainer.replay.sample(cfg.train.batch_size)

        wm_metrics = trainer.update_world_model(batch, step)
        wm_losses.append(wm_metrics.total_loss.item())

        a_loss, c_loss = trainer.update_actor_critic(batch, step)
        actor_losses.append(float(a_loss))
        critic_losses.append(float(c_loss))

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
    }


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
    results = [run_training(seed, steps=300) for seed in seeds]

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

    mean_kl = np.mean(kl_vals)
    print("Action KL mean:", mean_kl)

    wm_ok = wm_cv < 5e-3
    critic_ok = 0.05 < critic_cv < 1.0
    actor_ok = 0.5 < actor_cv < 3.0

    cfg = make_default_config()
    if cfg.world.num_aux_reward_heads > 0:
        kl_ok = mean_kl < 6.0
    else:
        kl_ok = 0.1 < mean_kl < 2.0

    assert wm_ok and critic_ok and actor_ok and kl_ok, "Statistical reproducibility FAILED"
