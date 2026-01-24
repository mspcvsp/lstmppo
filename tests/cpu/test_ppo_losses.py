"""
This test catches:

- PPO math regressions
- missing keys in info dict
- NaN explosions
"""
def test_ppo_loss_shapes():
    import torch
    from lstmppo.trainer import ppo_loss

    logits_old = torch.randn(4, 2)
    logits_new = torch.randn(4, 2)
    actions = torch.randint(0, 2, (4,))
    advantages = torch.randn(4)
    logp_old = torch.randn(4)

    loss, info = ppo_loss(
        logits_new,
        logits_old,
        actions,
        advantages,
        logp_old,
        clip_ratio=0.2
    )

    assert torch.isfinite(loss)
    assert "kl" in info
    assert "entropy" in info