"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Drift–saturation correlation is a *model‑level interpretability invariant*. It validates that as the LSTM’s
hidden‑state drift increases over time, its gate saturation also increases — a property that emerges from the
true GateLSTMCell dynamics:

    • correct gate wiring (i, f, g, o)
    • encoder → LSTM integration
    • hidden‑state update equations
    • long‑horizon stability
    • meaningful gate activations

Any fake state, fake policy, or synthetic rollout would bypass the real LSTM math and destroy the interpretability
signal this test is designed to catch.

This is a sentinel test for LSTM interpretability and gate correctness. Do not replace the real model here.
"""

import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput


def test_drift_saturation_correlation(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B = 3
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    # Long unroll to expose drift and saturation trends
    obs = torch.randn(B, 50, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # Drift: squared hidden gate magnitude over units and batch
    drift = out.gates.h_gates.pow(2).mean(dim=(0, 2))  # (T,)

    # Saturation: how close input gate is to 0 or 1
    sat = torch.minimum(out.gates.i_gates, 1 - out.gates.i_gates).mean(dim=(0, 2))  # (T,)

    # Correlation between drift and *negative* saturation (more drift → more saturation)
    corr = torch.corrcoef(torch.stack([drift, -sat]))[0, 1]

    assert corr > 0
