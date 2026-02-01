"""
Ensures drift variance across time is stable and finite.This test catches
pathological oscillations in hidden-state drift.
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_drift_variance_stability(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3
    trainer_state.cfg.trainer.debug_mode = True

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 100
    H = trainer_state.cfg.lstm.lstm_hidden_size
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    drift_t = out.gates.h_gates.pow(2).mean(dim=(0, 2))  # (T,)
    var = drift_t.var()

    # Variance should be finite and not extreme
    assert torch.isfinite(var)
    assert var < 0.01
