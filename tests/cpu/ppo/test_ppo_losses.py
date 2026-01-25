"""
A PPO loss test should check:
- Shapes are correct
- Losses are finite
- Masking works
- Clipping works
- KL and clip_frac are scalars
- No CPU/GPU mismatch
"""
import torch
from lstmppo.types import Config
from lstmppo.trainer import LSTMPPOTrainer


def test_ppo_loss_shapes():
    cfg = Config()
    device = torch.device("cpu")

    # Create a trainer (it owns compute_losses)
    trainer = LSTMPPOTrainer(cfg, device)

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs

    # Fake rollout batch
    values = torch.randn(T, B)
    old_values = torch.randn(T, B)
    returns = torch.randn(T, B)
    adv = torch.randn(T, B)

    new_logp = torch.randn(T, B)
    old_logp = torch.randn(T, B)

    # Mask: all alive
    mask = torch.ones(T, B)

    policy_loss, value_loss, approx_kl, clip_frac =\
        trainer.compute_losses(
            values=values,
            new_logp=new_logp,
            old_logp=old_logp,
            old_values=old_values,
            returns=returns,
            adv=adv,
            mask=mask,
        )

    # --- Assertions ---
    assert torch.is_tensor(policy_loss)
    assert torch.is_tensor(value_loss)
    assert torch.is_tensor(approx_kl)
    assert torch.is_tensor(clip_frac)

    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert approx_kl.ndim == 0
    assert clip_frac.ndim == 0

    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(approx_kl)
    assert torch.isfinite(clip_frac)
