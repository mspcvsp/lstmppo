"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate entropy is a fundamental interpretability signal. For sigmoid gates
(i, f, o), entropy is highest when activations are near 0.5 and decreases
as gates saturate toward {0, 1}. Scaling the input should push gates
toward saturation, thereby reducing entropy.

This test validates:
    • correct sigmoid gate behavior
    • encoder → LSTM integration
    • numerically stable entropy computation
    • meaningful gate activations under input scaling

Any fake state, fake policy, or synthetic rollout would bypass the real
LSTM math and invalidate this interpretability signal.

This is a sentinel test for gate‑entropy correctness. Do not replace the
real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_gate_entropy_decreases_with_saturation(trainer_state: TrainerState):
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

    i1 = out1.gates.i_gates
    i2 = out2.gates.i_gates

    eps = 1e-8

    # Binary entropy of sigmoid gate activations
    ent1 = -(i1 * torch.log(i1 + eps) + (1 - i1) * torch.log(1 - i1 + eps)).mean()
    ent2 = -(i2 * torch.log(i2 + eps) + (1 - i2) * torch.log(1 - i2 + eps)).mean()

    # Extreme inputs → more saturation → lower entropy
    assert ent2 < ent1
