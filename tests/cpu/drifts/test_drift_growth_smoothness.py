"""
Ensures drift grows smoothly with sequence length — no sudden jumps or 
collapses. This test ensures drift doesn’t explode or collapse between 
adjacent horizons.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput


def test_drift_growth_smoothness():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size

    lengths = [20, 40, 60, 80]
    num_samples = 20

    avg_drifts = []

    for L in lengths:
        drifts = []
        for _ in range(num_samples):
            obs = torch.randn(B, L, cfg.env.flat_obs_dim)
            h0 = torch.zeros(B, H)
            c0 = torch.zeros(B, H)
            out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
            drifts.append(out.gates.h_gates.pow(2).mean())
        avg_drifts.append(torch.stack(drifts).mean())

    eps = 1e-5
    for i in range(len(avg_drifts) - 1):
        assert avg_drifts[i+1] + eps >= avg_drifts[i] - eps
        assert (avg_drifts[i+1] - avg_drifts[i]).abs() < 0.01
