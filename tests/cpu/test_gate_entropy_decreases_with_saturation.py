"""
Ensures entropy decreases when gates saturate.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_gate_entropy_decreases_with_saturation():
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
    ent1 = -(out1.gates.i_gates * torch.log(out1.gates.i_gates + 1e-8)).mean()

    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))
    ent2 = -(out2.gates.i_gates * torch.log(out2.gates.i_gates + 1e-8)).mean()

    assert ent2 < ent1