"""
This test catches:

- accidental shape changes
- wrong flattening
- wrong output dimension
"""
import torch
from lstmppo.types import Config
from lstmppo.policy import LSTMPPOPolicy


def test_policy_forward_shapes():
    
    cfg = Config()

    model = LSTMPPOPolicy(cfg)
    obs = torch.randn(3, 4)

    logits = model(obs)

    assert logits.shape == (3, 2)