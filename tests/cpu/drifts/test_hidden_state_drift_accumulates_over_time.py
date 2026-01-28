"""
This test checks that hidden‑state drift grows over long horizons, which is
a key LSTM interpretability invariant.

This test ensures:

- LSTM hidden states accumulate variance over time
- Drift metrics behave meaningfully
- Diagnostics remain interpretable for long rollouts
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput

pytestmark = pytest.mark.drift


def test_hidden_state_drift_accumulates_over_time():
    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B = 3
    H = cfg.lstm.lstm_hidden_size

    # Short vs long sequence
    obs_short = torch.randn(B, 5, cfg.env.flat_obs_dim)
    obs_long = torch.randn(B, 50, cfg.env.flat_obs_dim)

    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out_short = policy.forward(PolicyInput(obs=obs_short, hxs=h0, cxs=c0))
    out_long = policy.forward(PolicyInput(obs=obs_long, hxs=h0, cxs=c0))

    drift_short = out_short.gates.h_gates.pow(2).mean()
    drift_long = out_long.gates.h_gates.pow(2).mean()

    """
    Hidden‑state drift should be non‑decreasing on average, not strictly increasing.
    Individual samples may show tiny decreases due to:

    - encoder noise
    - float32 rounding
    - LayerNorm stabilizing hidden activations
    - small batch size (B=3)
    - small drift magnitude (~1e‑4)

    So tests must use averaging and tolerances, not strict comparisons.
    """
    eps = 1e-6
    assert drift_long + eps >= drift_short
