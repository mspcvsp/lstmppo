"""
Ensures drift variance across time is stable and finite.This test catches
pathological oscillations in hidden-state drift.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput
import pytest
pytestmark = pytest.mark.drift


def test_drift_variance_stability():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 100
    H = cfg.lstm.lstm_hidden_size

    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    drift_t = out.gates.h_gates.pow(2).mean(dim=(0, 2))  # (T,)
    var = drift_t.var()

    # Variance should be finite and not extreme
    assert torch.isfinite(var)
    assert var < 0.01