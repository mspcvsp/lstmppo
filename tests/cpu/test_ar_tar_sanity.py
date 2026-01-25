"""
Ensures AR/TAR are scalars and always ≥ 0.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_ar_tar_sanity():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.lstm.lstm_ar_coef = 0.1
    cfg.lstm.lstm_tar_coef = 0.1

    policy = LSTMPPOPolicy(cfg)

    B, T = 3, 6
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    assert out.ar_loss.dim() == 0
    assert out.tar_loss.dim() == 0
    assert out.ar_loss >= 0
    assert out.tar_loss >= 0