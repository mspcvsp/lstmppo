"""
Over longer horizons, the LSTM cell state accumulates more magnitude/drift
than over short horizons. This test is important because the cell state 
c_t is the true “memory” of the LSTM, and drift accumulation is a strong 
signal that your diagnostics are wired correctly.

This test protects a subtle but crucial invariant:
-------------------------------------------------
- The LSTM cell state is the long‑term memory carrier
- Over longer sequences, it should accumulate more variance
- If it doesn’t, something is wrong with:
    - gate wiring
    - detach logic
    - LSTM unroll
    - diagnostics capture
    - or the encoder path

This test catches regressions that no other test will.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

def test_cell_state_drift_accumulates_over_time():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True   # disable DropConnect randomness

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size

    # Short vs long sequences
    obs_short = torch.randn(B, 5, cfg.env.flat_obs_dim)
    obs_long  = torch.randn(B, 50, cfg.env.flat_obs_dim)

    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    # Forward passes
    out_short = policy.forward(PolicyInput(obs=obs_short, hxs=h0, cxs=c0))
    out_long  = policy.forward(PolicyInput(obs=obs_long,  hxs=h0, cxs=c0))

    # Cell-state drift metric: mean squared magnitude of c_t over time
    drift_short = out_short.gates.c_gates.pow(2).mean()
    drift_long  = out_long.gates.c_gates.pow(2).mean()

    # Longer horizon → more accumulated drift
    assert drift_long > drift_short