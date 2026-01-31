"""
Ensures every gate tensor is (B, T, H) and stays batch‑first
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
    trainer_state.cfg.lstm.enc_hidden_size = 8
    trainer_state.cfg.lstm.lstm_hidden_size = 6

    policy = LSTMPPOPolicy(trainer_state)

    B, T = 3, 5
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    gates = out.gates
    for g in [gates.i_gates, gates.f_gates, gates.g_gates, gates.o_gates, gates.h_gates, gates.c_gates]:
        assert g.shape == (B, T, trainer_state.cfg.lstm.lstm_hidden_size)
