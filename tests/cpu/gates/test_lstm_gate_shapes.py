"""
Ensures every gate tensor is (B, T, H) and stays batch‑first
"""
import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

pytestmark = pytest.mark.gates


def test_lstm_gate_shapes():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.lstm.enc_hidden_size = 8
    cfg.lstm.lstm_hidden_size = 6

    policy = LSTMPPOPolicy(cfg)

    B, T = 3, 5
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    gates = out.gates
    for g in [gates.i_gates, gates.f_gates, gates.g_gates,
              gates.o_gates, gates.h_gates, gates.c_gates]:
        assert g.shape == (B, T, cfg.lstm.lstm_hidden_size)
