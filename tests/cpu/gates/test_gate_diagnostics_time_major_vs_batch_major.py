"""
Ensures diagnostics match regardless of input layout.
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
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    obs_tm = obs.transpose(0, 1)  # (T, B, F)

    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    # Batch-major
    out_bm = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # Time-major → convert to batch-major inside
    out_tm = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs_tm,
            actions=torch.zeros(T, B, dtype=torch.long),
            hxs=h0,
            cxs=c0,
        )
    )

    # Compare gate tensors (transpose time-major to batch-major)
    for g_bm, g_tm in [
        (out_bm.gates.i_gates, out_tm.gates.i_gates.transpose(0, 1)),
        (out_bm.gates.f_gates, out_tm.gates.f_gates.transpose(0, 1)),
        (out_bm.gates.g_gates, out_tm.gates.g_gates.transpose(0, 1)),
        (out_bm.gates.o_gates, out_tm.gates.o_gates.transpose(0, 1)),
        (out_bm.gates.h_gates, out_tm.gates.h_gates.transpose(0, 1)),
        (out_bm.gates.c_gates, out_tm.gates.c_gates.transpose(0, 1)),
    ]:
        assert torch.allclose(g_bm, g_tm)
