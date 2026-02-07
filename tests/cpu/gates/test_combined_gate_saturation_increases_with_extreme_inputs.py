"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate saturation under extreme inputs is a *model‑level interpretability
invariant* of the GateLSTMCell. When the magnitude of the input increases,
both sigmoid gates (i, f, o) and tanh‑based gates (g) should move closer
to their saturation regimes:

    • sigmoid → {0, 1}
    • tanh → {−1, +1}

This behavior emerges only from the true LSTM dynamics:
    • correct gate wiring (i, f, g, o)
    • encoder → LSTM integration
    • meaningful gate activations
    • stable long‑horizon behavior

Any fake state, fake policy, or synthetic rollout would bypass the real
LSTM math and invalidate this interpretability signal.

This is a sentinel test for LSTM gate correctness. Do not replace the
real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_combined_gate_saturation_increases_with_extreme_inputs(trainer_state: TrainerState):
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

    # Baseline vs extreme inputs
    out1 = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))

    gates1 = out1.gates
    gates2 = out2.gates

    # Sigmoid saturation: min(g, 1-g)
    sig1 = torch.minimum(gates1.i_gates, 1 - gates1.i_gates)
    sig2 = torch.minimum(gates2.i_gates, 1 - gates2.i_gates)

    # Tanh saturation: 1 - |g|
    tanh1 = 1 - gates1.g_gates.abs()
    tanh2 = 1 - gates2.g_gates.abs()

    # Combined saturation metric
    sat1 = torch.cat([sig1, tanh1], dim=-1).mean()
    sat2 = torch.cat([sig2, tanh2], dim=-1).mean()

    # Extreme inputs → more saturation → lower sat metric
    assert sat2 < sat1
