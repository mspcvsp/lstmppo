"""
Reusable FakeTrainerState for tests.

FakeState implements TrainerStateProtocol, which describes the minimal
interface required by LSTMPPOPolicy and RecurrentRolloutBuffer.
"""

from types import SimpleNamespace
from typing import Any, Protocol

import torch.nn as nn


class TrainerStateProtocol(Protocol):
    cfg: Any
    env_info: Any
    flat_obs_dim: int
    lstm: nn.Module


class FakeState:
    """Concrete implementation of TrainerStateProtocol for tests."""

    def __init__(
        self,
        cfg: SimpleNamespace,
        env_info: SimpleNamespace,
        flat_obs_dim: int,
        lstm: nn.Module,
    ):
        self.cfg = cfg
        self.env_info = env_info
        self.flat_obs_dim = flat_obs_dim
        self.lstm = lstm


def make_fake_state(
    rollout_steps: int = 8,
    num_envs: int = 2,
    obs_dim: int = 4,
    hidden_size: int = 4,
) -> TrainerStateProtocol:
    buffer_config = SimpleNamespace(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        mini_batch_envs=num_envs,
        lstm_hidden_size=hidden_size,
    )

    lstm_cfg = SimpleNamespace(
        lstm_ar_coef=0.0,
        lstm_tar_coef=0.0,
        lstm_dropout=0.0,
        lstm_layer_norm=False,
        enc_hidden_size=128,
        lstm_hidden_size=hidden_size,
        dropconnect_p=0.0,
    )

    trainer_cfg = SimpleNamespace(debug_mode=False)

    cfg = SimpleNamespace(
        buffer_config=buffer_config,
        lstm=lstm_cfg,
        trainer=trainer_cfg,
    )

    dummy_obs_space = SimpleNamespace(shape=(obs_dim,))
    env_info = SimpleNamespace(
        action_dim=1,
        obs_dim=obs_dim,
        obs_space=dummy_obs_space,
        flat_obs_dim=obs_dim,
    )

    lstm = nn.LSTM(
        input_size=obs_dim,
        hidden_size=hidden_size,
        batch_first=True,
    )

    return FakeState(cfg, env_info, obs_dim, lstm)
