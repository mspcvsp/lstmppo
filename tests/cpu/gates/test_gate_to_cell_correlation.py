"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
The forget gate (f_t) and the cell state magnitude (|c_t|) should not
exhibit a strong linear correlation. This is a *non‑relationship
invariant* of LSTM dynamics.

Reasons:
    • f_t is initialized near 0.73 (bias = +1.0), independent of c_t
    • c_t is an accumulator and grows with drift, noise, and gating
    • LayerNorm stabilizes h_t but not c_t
    • The LSTM equations do not impose a monotonic or sign‑stable
      relationship between f_t and |c_t|
    • Small batch sizes and short sequences introduce noise

Therefore:
    • The sign of the correlation is unstable
    • The magnitude should remain small
    • A strong positive or negative correlation indicates broken gating

This is a sentinel test for LSTM gate correctness. Do not replace the
real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_gate_to_cell_correlation(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 30
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # Flatten across (B, T)
    f = out.gates.f_gates.reshape(-1)
    c = out.gates.c_gates.abs().reshape(-1)

    corr = torch.corrcoef(torch.stack([f, c]))[0, 1]

    # The correlation should be small in magnitude
    assert corr.abs() < 0.2
