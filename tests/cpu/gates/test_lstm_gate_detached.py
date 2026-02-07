"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate tensors (i, f, g, o, c, h) are *diagnostic outputs*. They must never
be part of the autograd graph. If they require gradients, PPO’s backward
pass would:

    • backprop through the LSTM unroll
    • accumulate massive memory usage
    • break rollout‑level detach invariants
    • corrupt policy gradients
    • destabilize training

Correct behavior:
    • gates are computed inside the LSTM cell
    • then immediately detached
    • then stored only for interpretability

Any fake policy or synthetic gate object would bypass this invariant.

This is a sentinel test for PPO safety and LSTM diagnostic correctness.
Do not replace the real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.gates


def test_lstm_gate_detached(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 2, 4
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    gates = out.gates

    # All gate tensors must be detached from the computation graph
    for g in [
        gates.i_gates,
        gates.f_gates,
        gates.g_gates,
        gates.o_gates,
        gates.h_gates,
        gates.c_gates,
    ]:
        assert g.requires_grad is False
