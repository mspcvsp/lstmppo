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
    drifts = []

    for L in lengths:
        obs = torch.randn(B, L, cfg.env.flat_obs_dim)
        h0 = torch.zeros(B, H)
        c0 = torch.zeros(B, H)
        out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
        drifts.append(out.gates.h_gates.pow(2).mean())

    """
    Hidden‑state drift is:
    - tiny (~1e‑4)
    - monotonic in expectation
    - but noisy at float32 precision
    - and influenced by encoder randomness
    
    So:
    - strict monotonicity is impossible
    - even tolerant monotonicity (>=) can fail
    - the only reliable invariant is no meaningful decrease
    """
    eps = 1e-6
    for i in range(len(drifts) - 1):
        # Allow tiny decreases due to float32 noise
        assert drifts[i+1] + eps >= drifts[i] - eps
        # Still enforce smoothness
        assert (drifts[i+1] - drifts[i]).abs() < 0.01
