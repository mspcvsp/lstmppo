"""
Ensures AR/TAR are scalars and always ≥ 0.
"""

import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput


def test_ar_tar_sanity(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3
    trainer_state.cfg.lstm.lstm_ar_coef = 0.1
    trainer_state.cfg.lstm.lstm_tar_coef = 0.1

    policy = LSTMPPOPolicy(trainer_state)

    B, T = 3, 6
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    h0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)
    c0 = torch.zeros(B, trainer_state.cfg.lstm.lstm_hidden_size)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    assert out.ar_loss.dim() == 0
    assert out.tar_loss.dim() == 0
    assert out.ar_loss >= 0
    assert out.tar_loss >= 0
