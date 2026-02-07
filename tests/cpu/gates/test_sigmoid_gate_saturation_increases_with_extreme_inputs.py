"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate saturation is a *mathematical interpretability invariant* of the
GateLSTMCell. Extreme inputs should push:

    • sigmoid gates (i, f, o) toward {0, 1}
    • tanh gates (g) toward {−1, +1}

Correct saturation metrics:
    • Sigmoid:  min(g, 1 - g)      (lower = more saturated)
    • Tanh:     1 - |g|            (lower = more saturated)

This behavior emerges only from the true LSTM dynamics:
    • correct gate wiring
    • encoder → LSTM integration
    • stable activation functions
    • meaningful gate responses to input scaling

Any fake policy or synthetic gate object would bypass the real LSTM math
and invalidate this interpretability signal.

This is a sentinel test for gate‑saturation correctness. Do not replace
the real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_gate_saturation_increases_with_extreme_inputs(trainer_state: TrainerState):
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

    # Baseline
    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    i1 = out1.gates.i_gates  # (B, T, H)

    # Extreme inputs
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))
    i2 = out2.gates.i_gates

    # Sigmoid saturation metric (lower = more saturated)
    sat1 = torch.minimum(i1, 1 - i1).mean()
    sat2 = torch.minimum(i2, 1 - i2).mean()

    # Extreme inputs → more saturation → LOWER sat metric
    assert sat2 < sat1
