"""
Ensures diagnostics match regardless of input layout.
"""
import torch
from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import Config, PolicyInput, PolicyEvalInput
import pytest
pytestmark = pytest.mark.gates


def test_gate_diagnostics_time_major_vs_batch_major():

    cfg = Config()
    cfg.env.flat_obs_dim = 4
    cfg.env.action_dim = 3
    cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(cfg)
    policy.eval()

    B, T = 3, 5
    obs = torch.randn(B, T, cfg.env.flat_obs_dim)
    obs_tm = obs.transpose(0, 1)  # (T, B, F)

    h0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, cfg.lstm.lstm_hidden_size)

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