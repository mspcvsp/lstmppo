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

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

pytestmark = pytest.mark.drift


def test_cell_state_drift_accumulates_over_time():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size

    lengths = [5, 50]
    num_samples = 20  # average over multiple sequences

    avg_drifts = []

    for L in lengths:
        drifts = []
        for _ in range(num_samples):
            obs = torch.randn(B, L, cfg.env.flat_obs_dim)
            h0 = torch.zeros(B, H)
            c0 = torch.zeros(B, H)
            out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
            drifts.append(out.gates.c_gates.pow(2).mean())
        avg_drifts.append(torch.stack(drifts).mean())

    eps = 1e-5
    assert avg_drifts[1] + eps >= avg_drifts[0] - eps
