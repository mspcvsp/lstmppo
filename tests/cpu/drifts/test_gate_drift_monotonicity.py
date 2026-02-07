"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate‑drift monotonicity is a *model‑level* invariant of the GateLSTMCell. When the magnitude of the hidden state update
increases (e.g., by scaling the input), the drift of the hidden‑gate activations should also increase.

This test validates:

    • correct gate wiring (i, f, g, o)
    • encoder → LSTM integration
    • hidden‑state update equations
    • meaningful gate activations under input scaling

Any fake state, fake policy, or synthetic rollout would bypass the true LSTM dynamics and invalidate the signal this
test is designed to catch. This is a sentinel test for LSTM interpretability and gate correctness. Do not replace the
real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_gate_drift_monotonicity(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 5
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    # Baseline drift
    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    drift1 = out1.gates.h_gates.pow(2).mean()

    # Increased input magnitude → increased drift
    out2 = policy.forward(PolicyInput(obs=obs * 3.0, hxs=h0, cxs=c0))
    drift2 = out2.gates.h_gates.pow(2).mean()

    assert drift2 > drift1
