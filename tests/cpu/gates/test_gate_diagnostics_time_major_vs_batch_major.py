"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Gate diagnostics must be invariant to input layout. Whether observations
arrive as batch‑major (B, T, F) or time‑major (T, B, F), the underlying
LSTM dynamics and gate activations must be identical after appropriate
transposition.

This invariant ensures:
    • correct encoder → LSTM integration
    • consistent gate extraction in both forward() and evaluate_actions_sequence()
    • correct time/batch transposition logic
    • stable diagnostics across rollout and evaluation paths
    • no silent shape‑related regressions

Any fake state, fake policy, or synthetic rollout would bypass the real
LSTM math and invalidate this consistency check.

This is a sentinel test for gate‑diagnostic correctness. Do not replace
the real model here.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyEvalInput, PolicyInput

pytestmark = pytest.mark.gates


def test_gate_diagnostics_time_major_vs_batch_major(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 5
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs_bm = torch.randn(B, T, D)
    obs_tm = obs_bm.transpose(0, 1)  # (T, B, F)

    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    # Batch‑major path
    out_bm = policy.forward(PolicyInput(obs=obs_bm, hxs=h0, cxs=c0))

    # Time‑major path (policy converts internally to batch‑major)
    out_tm = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs_tm,
            actions=torch.zeros(T, B, dtype=torch.long),
            hxs=h0,
            cxs=c0,
        )
    )

    # Compare gate tensors (transpose time‑major to batch‑major)
    for g_bm, g_tm in [
        (out_bm.gates.i_gates, out_tm.gates.i_gates.transpose(0, 1)),
        (out_bm.gates.f_gates, out_tm.gates.f_gates.transpose(0, 1)),
        (out_bm.gates.g_gates, out_tm.gates.g_gates.transpose(0, 1)),
        (out_bm.gates.o_gates, out_tm.gates.o_gates.transpose(0, 1)),
        (out_bm.gates.h_gates, out_tm.gates.h_gates.transpose(0, 1)),
        (out_bm.gates.c_gates, out_tm.gates.c_gates.transpose(0, 1)),
    ]:
        assert torch.allclose(g_bm, g_tm)
