import pytest

from dreamerrl.eval.popgym_eval import (
    train_popgym_seed,
)
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


@pytest.mark.manual
@pytest.mark.popgym_learning
def test_repeat_previous_easy_learning():
    cfg = make_default_config()
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"

    # PopGym-specific DreamerV3 tuning
    cfg.world.deter_size = 32
    cfg.world.stoch_size = 8
    cfg.world.num_classes = 8
    cfg.world.hidden_size = 64

    cfg.world.imagination_horizon = 2
    cfg.world.kl_scale = 0.005
    cfg.world.free_nats = 0.5

    cfg.train.model_lr = 3e-5
    cfg.train.grad_clip = 10.0
    cfg.train.disable_aux_losses = True
    cfg.train.use_amp = False

    steps = 5000
    trainer = DreamerTrainer(cfg)
    summary = train_popgym_seed(trainer, steps)

    assert summary["mean_return"] > 2.0
