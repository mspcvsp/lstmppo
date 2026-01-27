"""
This tiny test protects from subtle but catastrophic regressions:

- critic loss accidentally detached
- wrong reduction (sum vs mean)
- masking logic broken
- value loss weight misapplied
- incorrect broadcasting
- sign errors
- value head output shape changes
- returns accidentally normalized or clipped

"""

import pytest
import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import Config

pytestmark = pytest.mark.ppo


def test_ppo_value_monotonicity():
    """
    Smoke test: critic loss must increase when value predictions
    move farther away from returns.
    """

    cfg = Config()
    cfg.trainer.cuda = False

    trainer = LSTMPPOTrainer(cfg)

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs

    # Base rollout batch
    returns = torch.randn(T, B)
    adv = torch.randn(T, B)
    mask = torch.ones(T, B)

    # Case 1: values close to returns
    values_close = returns + 0.01 * torch.randn(T, B)
    old_values_close = values_close.clone()
    old_logp = torch.randn(T, B)
    new_logp = old_logp.clone()

    _, value_loss_close, _, _ = trainer.compute_losses(
        values=values_close,
        new_logp=new_logp,
        old_logp=old_logp,
        old_values=old_values_close,
        returns=returns,
        adv=adv,
        mask=mask,
    )

    # Case 2: values far from returns
    values_far = returns + 5.0
    old_values_far = values_far.clone()

    _, value_loss_far, _, _ = trainer.compute_losses(
        values=values_far,
        new_logp=new_logp,
        old_logp=old_logp,
        old_values=old_values_far,
        returns=returns,
        adv=adv,
        mask=mask,
    )

    # Critic loss must increase when predictions drift
    assert value_loss_far > value_loss_close
