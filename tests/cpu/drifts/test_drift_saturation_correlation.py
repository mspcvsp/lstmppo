"""
Ensures that when drift increases, saturation also increases
(a deep interpretability invariant).
"""
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput


def test_drift_saturation_correlation():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size

    obs = torch.randn(B, 50, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    drift = out.gates.h_gates.pow(2).mean(dim=(0, 2))  # (T,)

    sat   = torch.minimum(
        out.gates.i_gates,
        1 - out.gates.i_gates
        ).mean(dim=(0, 2))  # (T,)

    corr = torch.corrcoef(torch.stack([drift, -sat]))[0, 1]

    assert corr > 0
