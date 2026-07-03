import torch

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def test_kl_dynamics_invariant():
    """
    KL must:
      • be finite
      • not collapse to zero
      • not explode
      • be in a reasonable Dreamer-V3 range
    """

    cfg = make_default_config()
    cfg.train.seed = 0
    cfg.train.cuda = False
    cfg.train.enforce_length_invariants = True

    cfg.env.num_envs = 1
    cfg.env.env_id = "popgym-RepeatFirstEasy-v0"
    cfg.env.max_episode_steps = 5
    cfg.train.collect_steps = cfg.env.max_episode_steps
    cfg.train.seq_len = cfg.env.max_episode_steps

    cfg.train.batch_size = 4
    cfg.world.num_aux_reward_heads = 0

    trainer = DreamerTrainer(cfg)

    kl_values = []

    for step in range(20):
        trainer.collect_env_steps()
        batch = trainer.replay.sample(cfg.train.batch_size)
        metrics = trainer.update_world_model(batch, step)
        kl_values.append(metrics.kl_dyn.item() + metrics.kl_rep.item())

    kl_tensor = torch.tensor(kl_values)

    # KL must be finite
    assert torch.isfinite(kl_tensor).all(), "KL contains NaN or Inf"

    # KL must not collapse
    assert kl_tensor.mean() > 0.01, "KL collapsed — posterior/prior identical"

    # KL must not explode
    assert kl_tensor.mean() < 20.0, "KL exploded — unstable RSSM"
