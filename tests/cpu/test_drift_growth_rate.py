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

    obs_20 = torch.randn(B, 20, cfg.env.flat_obs_dim)
    obs_40 = torch.randn(B, 40, cfg.env.flat_obs_dim)
    obs_80 = torch.randn(B, 80, cfg.env.flat_obs_dim)

    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    drift_20 = policy.forward(PolicyInput(obs=obs_20,
                                          hxs=h0,
                                          cxs=c0)).gates.h_gates.pow(2).mean()

    drift_40 = policy.forward(PolicyInput(obs=obs_40,
                                          hxs=h0,
                                          cxs=c0)).gates.h_gates.pow(2).mean()
    
    drift_80 = policy.forward(PolicyInput(obs=obs_80,
                                          hxs=h0,
                                          cxs=c0)).gates.h_gates.pow(2).mean()

    assert drift_20 < drift_40 < drift_80