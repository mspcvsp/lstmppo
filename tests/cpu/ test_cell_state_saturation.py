"""
Ensures cell state saturates more under extreme inputs.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_cell_state_saturation():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 20
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))

    c1 = out1.gates.c_gates
    c2 = out2.gates.c_gates

    sat1 = (1 - c1.tanh().abs()).mean()
    sat2 = (1 - c2.tanh().abs()).mean()

    assert sat2 < sat1