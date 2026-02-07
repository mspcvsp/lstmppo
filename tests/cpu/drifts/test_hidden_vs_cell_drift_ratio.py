"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
The LSTM cell state (c_t) is an accumulator:
    c_t = f_t * c_{t-1} + i_t * g_t

The hidden state (h_t) is a squashed, gated projection of the cell state:
    h_t = o_t * tanh(c_t)

Because:
    • tanh compresses large magnitudes
    • o_t (sigmoid) further scales activations
    • LayerNorm stabilizes hidden activations

…the hidden state cannot drift as fast as the cell state.
The correct invariant is: **cell drift ≥ hidden drift**.

This relationship emerges only from the true LSTM dynamics:
    • correct gate wiring (i, f, g, o)
    • stable accumulator behavior in c_t
    • meaningful hidden‑state gating
    • correct encoder → LSTM integration

Any fake state, fake policy, or synthetic rollout would bypass the real
LSTM math and invalidate this interpretability signal.

This is a sentinel test for LSTM correctness. Do not replace the real model.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_hidden_vs_cell_drift_ratio(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 40
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # Drift magnitudes
    h_drift = out.gates.h_gates.pow(2).mean()
    c_drift = out.gates.c_gates.pow(2).mean()

    # Cell drift should be ≥ hidden drift
    assert c_drift > h_drift
