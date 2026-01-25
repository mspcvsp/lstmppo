"""
Ensures gate tensors are not part of the computation graph.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput
import pytest
pytestmark = pytest.mark.gates


def test_lstm_gate_detached():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3

    policy = LSTMPPOPolicy(cfg)

    B, T = 2, 4
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    gates = out.gates
    for g in [gates.i_gates, gates.f_gates, gates.g_gates,
              gates.o_gates, gates.h_gates, gates.c_gates]:
        assert g.requires_grad is False