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
    """
    Fast functional PopGym integration test.
    Confirms Dreamer can train and metrics are sane.
    NOT a reproducibility test.
    """

    if not request.config.getoption("--run-manual"):
        pytest.skip("Manual test skipped. Use --run-manual to enable.")

    env_id = request.config.getoption("--env")
    steps = int(request.config.getoption("--steps") or 1000)

    seeds = [0, 1, 2]
    results = []

    for seed in seeds:
        cfg = make_default_config()
        cfg.env.env_id = env_id
        cfg.env.seed = seed
        cfg.train.seed = seed

        # PopGym tests should be FAST, not deterministic
        cfg.env.parallel = False  # SyncVectorEnv for stability
        cfg.train.deterministic_env = False
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

    # Functional sanity checks (not reproducibility)
    assert summary["wm_cv"] < 5e-2
    assert 0.1 < summary["actor_cv"] < 10.0
    assert 0.01 < summary["critic_cv"] < 5.0
    assert 0.01 < summary["action_kl"] < 5.0

    # Learning signal: > 0.1 is enough for RepeatPreviousEasy
    assert summary["mean_return"] > 0.1
