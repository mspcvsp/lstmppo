import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput
import pytest
pytestmark = pytest.mark.gates

def test_combined_gate_saturation_increases_with_extreme_inputs():
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

    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))

    gates1 = out1.gates
    gates2 = out2.gates

    # Sigmoid gates
    sig1 = torch.minimum(gates1.i_gates, 1 - gates1.i_gates)
    sig2 = torch.minimum(gates2.i_gates, 1 - gates2.i_gates)

    # Tanh gates
    tanh1 = 1 - gates1.g_gates.abs()
    tanh2 = 1 - gates2.g_gates.abs()

    sat1 = torch.cat([sig1, tanh1], dim=-1).mean()
    sat2 = torch.cat([sig2, tanh2], dim=-1).mean()

    assert sat2 < sat1