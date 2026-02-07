"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Hidden‑state drift accumulation is a *core interpretability invariant* of
the GateLSTMCell. Over longer horizons, the LSTM hidden state should
accumulate more variance and exhibit larger drift magnitudes. This
emerges only from the true LSTM dynamics:

    • correct gate wiring (i, f, g, o)
    • encoder → LSTM integration
    • stable long‑horizon unrolls
    • meaningful hidden‑state updates
    • correct detach and state‑flow logic

Any fake state, fake policy, or synthetic rollout would bypass the real
LSTM math and invalidate the signal this test is designed to catch.

This is a sentinel test for long‑sequence LSTM interpretability and
stability. Do not replace the real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_hidden_state_drift_accumulates_over_time(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B = 3
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    # Short vs long sequences
    obs_short = torch.randn(B, 5, D)
    obs_long = torch.randn(B, 50, D)

    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out_short = policy.forward(PolicyInput(obs=obs_short, hxs=h0, cxs=c0))
    out_long = policy.forward(PolicyInput(obs=obs_long, hxs=h0, cxs=c0))

    drift_short = out_short.gates.h_gates.pow(2).mean()
    drift_long = out_long.gates.h_gates.pow(2).mean()

    # Hidden‑state drift should be non‑decreasing on average.
    # Tiny decreases are allowed due to:
    #   • encoder noise
    #   • float32 rounding
    #   • LayerNorm stabilization
    #   • small batch size (B=3)
    #   • small drift magnitude (~1e‑4)
    eps = 1e-6
    assert drift_long + eps >= drift_short
