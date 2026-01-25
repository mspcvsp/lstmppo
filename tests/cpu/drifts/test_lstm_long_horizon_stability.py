"""
Ensures the LSTM does not explode over long sequences.
"""
import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

pytestmark = pytest.mark.drift


def test_lstm_long_horizon_stability():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 200
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    h = out.gates.h_gates
    c = out.gates.c_gates

    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()
    assert h.abs().max() < 50
    assert c.abs().max() < 50
