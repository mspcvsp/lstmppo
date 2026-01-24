"""
This test catches:

- accidental shape changes
- wrong flattening
- wrong output dimension
"""
def test_policy_forward_shapes():
    import torch
    from lstmppo.policy import PolicyNetwork

    model = PolicyNetwork(obs_dim=4, act_dim=2)
    obs = torch.randn(3, 4)

    logits = model(obs)

    assert logits.shape == (3, 2)