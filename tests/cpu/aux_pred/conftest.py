from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch.nn as nn


@dataclass
class FakeState:
    cfg: SimpleNamespace
    env_info: SimpleNamespace
    flat_obs_dim: int
    lstm: nn.Module


@pytest.fixture
def fake_state() -> FakeState:
    rollout_steps = 8
    num_envs = 2
    obs_dim = 4
    hidden_size = 4

    buffer_config = SimpleNamespace(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        mini_batch_envs=num_envs,
        lstm_hidden_size=hidden_size,
    )

    lstm_cfg = SimpleNamespace(
        enc_hidden_size=128,
        lstm_hidden_size=128,
        lstm_ar_coef=1.0,
        lstm_tar_coef=0.5,
        dropconnect_p=0.0,
        lstm_layer_norm=False,
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

    trainer_cfg = SimpleNamespace(
        debug_mode=False,
    )

    cfg = SimpleNamespace(
        buffer_config=buffer_config,
        lstm=lstm_cfg,
        trainer=trainer_cfg,
    )

    return FakeState(
        cfg=cfg,
        env_info=env_info,
        flat_obs_dim=obs_dim,
        lstm=lstm,
    )
