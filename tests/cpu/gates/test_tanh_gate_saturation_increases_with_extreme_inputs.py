"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
The g‑gate of an LSTM uses tanh activation. Its saturation behavior is a
mathematical invariant:

    • tanh(x) → ±1 as |x| → ∞
    • saturation metric = 1 - |g|
      (lower = more saturated)

Extreme inputs should push g‑gates toward ±1, reducing the saturation
metric. This behavior emerges only from the true LSTM dynamics:

    • correct g‑gate wiring
    • encoder → LSTM integration
    • stable tanh activation
    • meaningful gate responses to input scaling

Any fake policy or synthetic gate object would bypass the real LSTM math
and invalidate this interpretability signal.

This is a sentinel test for tanh‑gate saturation correctness. Do not
replace the real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_tanh_gate_saturation_increases_with_extreme_inputs(trainer_state: TrainerState):
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
    g1 = out1.gates.g_gates

    # Extreme inputs
    out2 = policy.forward(PolicyInput(obs=obs * 5.0, hxs=h0, cxs=c0))
    g2 = out2.gates.g_gates

    # Tanh saturation metric (lower = more saturated)
    sat1 = (1 - g1.abs()).mean()
    sat2 = (1 - g2.abs()).mean()

    # Extreme inputs → more saturation → LOWER sat metric
    assert sat2 < sat1
