import torch
from lstmppo.types import Config
from lstmppo.trainer import LSTMPPOTrainer


def test_ppo_kl_watchdog():
    """
    Smoke test: PPO must detect synthetic KL drift.
    If new_logp is shifted away from old_logp, approx_kl must increase.
    """

    cfg = Config()
    device = torch.device("cpu")

    trainer = LSTMPPOTrainer(cfg, device)

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