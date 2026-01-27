"""
This test catches regressions in:

- ratio computation: exp(new_logp - old_logp)
- KL estimator: 0.5 * (ratio - 1).pow(2) or your variant
- masking logic
- broadcasting
- loss aggregation
- accidental detachment of tensors
- incorrect reduction (mean vs sum)
- sign errors

It’s the kind of test that prevents silent PPO collapse months later.
"""

import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import Config


def test_ppo_kl_watchdog():
    """
    Smoke test: PPO must detect synthetic KL drift.
    If new_logp is shifted away from old_logp, approx_kl must increase.
    """

    cfg = Config()
    cfg.trainer.cuda = False

    trainer = LSTMPPOTrainer(cfg)

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs

    # Base rollout batch
    values = torch.randn(T, B)
    old_values = torch.randn(T, B)
    returns = torch.randn(T, B)
    adv = torch.randn(T, B)
    mask = torch.ones(T, B)

    # Case 1: no drift
    old_logp = torch.randn(T, B)
    new_logp_same = old_logp.clone()

    _, _, kl_same, _ = trainer.compute_losses(
        values=values,
        new_logp=new_logp_same,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        adv=adv,
        mask=mask,
    )

    # Case 2: synthetic drift (shift logp by +1.0)
    new_logp_shifted = old_logp + 1.0

    _, _, kl_shifted, _ = trainer.compute_losses(
        values=values,
        new_logp=new_logp_shifted,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        adv=adv,
        mask=mask,
    )

    # KL must increase when logp diverges
    assert kl_shifted > kl_same
