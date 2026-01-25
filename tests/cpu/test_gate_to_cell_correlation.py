"""
Ensures forget gate correlates with cell‑state magnitude
(core LSTM behavior).
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_gate_to_cell_correlation():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 30
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    f = out.gates.f_gates.reshape(-1)
    c = out.gates.c_gates.abs().reshape(-1)

    corr = torch.corrcoef(torch.stack([f, c]))[0, 1]

    # Forget gate should positively correlate with cell magnitude
    assert corr > 0