"""
Ensures entropy decreases when gates saturate.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_gate_entropy_decreases_with_saturation():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 5
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))

    i1 = out1.gates.i_gates
    i2 = out2.gates.i_gates

    eps = 1e-8
    
    ent1 = (
        -(i1 * torch.log(i1 + eps) +
          (1 - i1) * torch.log(1 - i1 + eps)).mean()
    )
    
    ent2 = (
        -(i2 * torch.log(i2 + eps) + 
          (1 - i2) * torch.log(1 - i2 + eps)).mean()
    )

    assert ent2 < ent1