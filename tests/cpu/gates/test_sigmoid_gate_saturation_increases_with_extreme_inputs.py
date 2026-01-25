"""
Instead of checking abs(gate) > 0.95, measure distance to the 
saturation boundaries:

For sigmoid gates (i, f, o):
---------------------------
- saturation happens when values approach 0 or 1

For tanh gates (g):
---------------------------
- saturation happens when values approach -1 or 1

So the correct saturation metric is:

Sigmoid saturation:
------------------
sat = min(gate, 1-gate) (Lower = more saturated)

Tanh saturation:
------------------
sat = 1 - |gate| (Lower = more saturated)
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput
import pytest
pytestmark = pytest.mark.gates


def test_gate_saturation_increases_with_extreme_inputs():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 5
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    # Baseline
    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    i1 = out1.gates.i_gates  # (B,T,H)

    # Extreme inputs
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))
    i2 = out2.gates.i_gates

    # Sigmoid saturation metric: closer to 0 or 1 = more saturated
    sat1 = torch.minimum(i1, 1 - i1).mean()
    sat2 = torch.minimum(i2, 1 - i2).mean()

    # More extreme inputs → more saturation → LOWER sat metric
    assert sat2 < sat1
