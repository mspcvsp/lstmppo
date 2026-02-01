"""
Ensures the LSTM does not explode over long sequences.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_lstm_long_horizon_stability(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3
    trainer_state.cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 200
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    h = out.gates.h_gates
    c = out.gates.c_gates

    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()
    assert h.abs().max() < 50
    assert c.abs().max() < 50
