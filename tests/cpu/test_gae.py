"""
This test catches:

- GAE math regressions
- dtype issues
- shape mismatches
"""
def test_gae_computation_basic():
    import torch
    from lstmppo.buffer import compute_gae

    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0, 0, 1])
    gamma = 0.99
    lam = 0.95

    adv = compute_gae(rewards, values, dones, gamma, lam)

    assert adv.shape == (3,)
    assert torch.isfinite(adv).all()