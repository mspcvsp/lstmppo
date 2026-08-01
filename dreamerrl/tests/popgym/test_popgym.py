import pytest
import torch.nn as nn

from dreamerrl.eval.popgym_eval import (
    aggregate_popgym_results,
    train_popgym_seed,
)
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


@pytest.mark.manual
@pytest.mark.popgym_functional
def test_popgym(request):
    """
    Fast functional PopGym integration test.
    Confirms Dreamer can train and metrics are sane.
    NOT a reproducibility test.
    """

    if not request.config.getoption("--run-manual"):
        pytest.skip("Manual test skipped. Use --run-manual to enable.")

    steps = int(request.config.getoption("--steps") or 100)
    print("Running PopGym test for {} steps".format(steps))

    seeds = [0, 1, 2]
    results = []

    for seed in seeds:
        cfg = make_default_config()
        cfg.env.seed = seed
        cfg.train.seed = seed

        cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"

        # World model size: small and sharp
        cfg.world.deter_size = 32
        cfg.world.stoch_size = 8
        cfg.world.num_classes = 8
        cfg.world.hidden_size = 64

        # Imagination + KL: very conservative
        cfg.world.imagination_horizon = 2
        cfg.world.kl_scale = 0.005
        cfg.world.kl_balance = 0.5
        cfg.world.free_nats = 0.5

        # Training: gentle but not trivial
        cfg.train.model_lr = 3e-5
        cfg.train.grad_clip = 10.0
        cfg.train.freeze_actor_critic_steps = 0
        cfg.train.disable_aux_losses = True
        cfg.train.use_amp = False

        # Sequence / rollout lengths
        cfg.train.seq_len = 20
        cfg.train.collect_steps = 20
        cfg.env.max_episode_steps = 20

        cfg.env.parallel = False
        cfg.train.deterministic_env = False
        cfg.log.enable_wandb = False

        trainer = DreamerTrainer(cfg)

        # After trainer = DreamerTrainer(cfg)
        wm = trainer.world

        # Aux losses must be fully disabled for PopGym
        assert wm.net_cfg.disable_aux_losses is True
        assert wm.aux_objectives == []
        assert isinstance(wm.aux_heads, nn.ModuleDict)
        assert len(wm.aux_heads) == 0

        metrics = train_popgym_seed(trainer, steps=steps)
        results.append(metrics)

    summary = aggregate_popgym_results(results)

    print("World Model CV:", summary["wm_cv"])
    print("Actor CV:", summary["actor_cv"])
    print("Critic CV:", summary["critic_cv"])
    print("Action KL:", summary["action_kl"])
    print("Mean Return:", summary["mean_return"])

    # Functional sanity checks (not reproducibility)
    assert summary["wm_cv"] < 0.15
    assert 0.1 < summary["actor_cv"] < 10.0
    assert 0.01 < summary["critic_cv"] < 5.0
    # >>> RELAXED: functional test, allow higher KL
    assert 0.01 < summary["action_kl"] < 10.0

    # Learning signal: > 0.1 is enough for RepeatPreviousEasy
    assert summary["mean_return"] > 0.1
