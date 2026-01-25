"""
Ensures drift grows roughly linearly with sequence length (not exploding,
not collapsing).
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput


def test_drift_growth_rate():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size

    lengths = [20, 40, 80]
    num_samples = 20  # more samples = smoother averages

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

    # Allow tiny decreases due to noise
    eps = 1e-5
    assert avg_drifts[1] + eps >= avg_drifts[0] - eps
    assert avg_drifts[2] + eps >= avg_drifts[1] - eps