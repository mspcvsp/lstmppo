"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate tensors must always have shape (B, T, H) in batch‑major format.
This is a *structural invariant* of the entire diagnostics pipeline.

If gate shapes drift from (B, T, H), the following break:
    • drift metrics (expect time‑major or batch‑major consistency)
    • saturation metrics
    • entropy metrics
    • correlation tests
    • long‑horizon stability diagnostics
    • PPO rollout and evaluation paths

Correct behavior:
    • forward() returns gates in (B, T, H)
    • evaluate_actions_sequence() returns gates in (T, B, H) but is
      transposed internally before being wrapped in LSTMGates
    • all gate tensors remain batch‑first inside the policy output

Any fake policy or synthetic gate object would bypass this invariant.

This is a sentinel test for gate‑shape correctness. Do not replace the
real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_lstm_gate_shapes(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    # Explicit small dims to make shape errors obvious
    trainer_state.cfg.lstm.enc_hidden_size = 8
    trainer_state.cfg.lstm.lstm_hidden_size = 6

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 5
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    gates = out.gates

    # All gate tensors must be (B, T, H)
    for g in [
        gates.i_gates,
        gates.f_gates,
        gates.g_gates,
        gates.o_gates,
        gates.h_gates,
        gates.c_gates,
    ]:
        assert g.shape == (B, T, H)
