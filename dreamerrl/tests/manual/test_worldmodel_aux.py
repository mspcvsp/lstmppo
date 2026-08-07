import pytest

from dreamerrl.tests.manual.utils import summarize_metrics
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import make_default_config


@pytest.mark.manual
@pytest.mark.worldmodel_aux
def test_worldmodel_aux_stability():
    seeds = [0, 1, 2]
    steps = 800
    freeze = steps  # actor/critic fully frozen

    results = []
    for seed in seeds:
        cfg = make_default_config()

        cfg.train.seed = seed
        cfg.train.disable_aux_losses = False  # AUX LOSSES ON
        cfg.train.freeze_actor_critic_steps = freeze
        cfg.train.deterministic_imagination = True
        cfg.train.deterministic_env = True
        cfg.train.enforce_length_invariants = True
        cfg.train.cuda = True

        cfg.env.env_id = "popgym-RepeatFirstEasy-v0"
        cfg.env.num_envs = 4
        cfg.env.deterministic = True
        cfg.env.seed = seed
        cfg.env.parallel = False

        cfg.train.collect_steps = 10
        cfg.env.max_episode_steps = cfg.train.collect_steps
        cfg.train.seq_len = cfg.train.collect_steps

        set_global_seeds(seed)
        trainer = DreamerTrainer(cfg)

        wm_metrics = []

        print("Warming up the environment for 200 steps...\n", flush=True)

        for _ in range(200):
            trainer.collect_env_steps()

        for step in range(steps):
            trainer.collect_env_steps()
            batch = trainer.replay.sample(cfg.train.batch_size)
            metrics = trainer.update_world_model(batch, step)

            if step % 50 == 0:
                print(
                    f"[aux-wm] seed={seed} step={step}/{steps} "
                    f"total={metrics.total_loss.item():.4f} "
                    f"recon={metrics.recon_loss.item():.4f} "
                    f"reward={metrics.reward_loss.item():.4f} "
                    f"cont={metrics.cont_loss.item():.4f} "
                    f"kl_dyn={metrics.kl_dyn.item():.4f} "
                    f"kl_rep={metrics.kl_rep.item():.4f}",
                    flush=True,
                )

            wm_metrics.append(metrics)

        results.append(wm_metrics)

    # Summaries
    summary = summarize_metrics(results)

    # Stability thresholds (tight but realistic)
    assert summary["total_loss"][2] < 0.05
    assert summary["recon_loss"][2] < 0.05
    assert summary["reward_loss"][2] < 0.10
    assert summary["cont_loss"][2] < 0.10
    assert summary["kl_dyn"][2] < 0.10
    assert summary["kl_rep"][2] < 0.10
