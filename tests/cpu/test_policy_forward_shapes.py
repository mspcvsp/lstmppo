"""
This test catches:

- accidental shape changes
- wrong flattening
- wrong output dimension
"""
import torch
from lstmppo.types import Config, PolicyInput, initialize_config
from lstmppo.policy import LSTMPPOPolicy


def test_policy_forward_shapes():
    
    cfg = Config()
    cfg = initialize_config(cfg)
    model = LSTMPPOPolicy(cfg)

    B = 3

    inp = PolicyInput(
        obs=torch.randn(B, cfg.obs_dim),
        hxs=torch.randn(B, cfg.lstm.lstm_hidden_size),
        cxs=torch.randn(B, cfg.lstm.lstm_hidden_size),
    )

    outp = model(inp)

    assert outp.logits.shape == (B, cfg.action_dim)
    assert outp.values.shape == (B,)
    assert outp.new_hxs.shape == (B, cfg.lstm.lstm_hidden_size)
    assert outp.new_cxs.shape == (B, cfg.lstm.lstm_hidden_size)
    assert outp.ar_loss.shape == torch.Size([])
    assert outp.tar_loss.shape == torch.Size([])

    for name in [      
        "i_gates", "f_gates", "o_gates",
        "o_gates", "c_gates", "h_gates"
        ]:
        
        t = getattr(outp.gates, name)
        assert t.shape == (B, 1, cfg.lstm.lstm_hidden_size)
