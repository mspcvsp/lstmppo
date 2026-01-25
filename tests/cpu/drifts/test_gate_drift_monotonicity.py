"""
Verifies gate drift increases when hidden state magnitude increases
"""
import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

pytestmark = pytest.mark.drift


def test_gate_drift_monotonicity():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3

    policy = LSTMPPOPolicy(cfg)

    B, T = 3, 5
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    # Baseline
    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    drift1 = out1.gates.h_gates.pow(2).mean()

    # Synthetic drift: scale obs
    out2 = policy.forward(PolicyInput(obs=obs * 3.0, hxs=h0, cxs=c0))
    drift2 = out2.gates.h_gates.pow(2).mean()

    assert drift2 > drift1
