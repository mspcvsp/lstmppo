import pytest

from dreamerrl.eval.popgym_eval import (
    aggregate_popgym_results,
    train_popgym_seed,
)
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


@pytest.mark.manual
@pytest.mark.popgym
def test_popgym(request):
    if not request.config.getoption("--run-manual"):
        pytest.skip("Manual test skipped. Use --run-manual to enable.")

    env_id = request.config.getoption("--env")
    steps = int(request.config.getoption("--steps"))

    seeds = [0, 1, 2]
    results = []

    for seed in seeds:
        cfg = make_default_config()
        cfg.env.env_id = env_id
        cfg.env.seed = seed
        cfg.train.seed = seed
        cfg.log.enable_wandb = False

        trainer = DreamerTrainer(cfg)
        metrics = train_popgym_seed(trainer, steps=steps)
        results.append(metrics)

    summary = aggregate_popgym_results(results)

    print("World Model CV:", summary["wm_cv"])
    print("Actor CV:", summary["actor_cv"])
    print("Critic CV:", summary["critic_cv"])
    print("Action KL:", summary["action_kl"])
    print("Mean Return:", summary["mean_return"])

    assert summary["wm_cv"] < 1e-3
    assert 0.5 < summary["actor_cv"] < 3.0
    assert 0.05 < summary["critic_cv"] < 1.0
    assert 0.1 < summary["action_kl"] < 2.0
    assert summary["mean_return"] > 0.8
