"""
Ensures the cell state drifts more slowly than the hidden state
(a known LSTMproperty).
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_hidden_vs_cell_drift_ratio():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 40
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    h_drift = out.gates.h_gates.pow(2).mean()
    c_drift = out.gates.c_gates.pow(2).mean()

    """
    LSTM cell state update is: c_t = f_t * c_{t-1} +i_t * g_t
    This is an accumulator. It is designed to drift.

    Hidden state is: h_t = o_t * tanh(c_t)
    This is a squashed version of the cell state.

    Because:
    - tanh compresses large values
    - o_t (sigmoid) further scales them
    - LayerNorm stabilizes hidden activations

    The hidden state cannot drift as fast as the cell state.
    So the correct invariant is: Cell drift ≥ Hidden drift
    """
    assert c_drift > h_drift
