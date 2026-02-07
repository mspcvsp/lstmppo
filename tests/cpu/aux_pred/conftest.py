from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@dataclass
class FakeState:
    cfg: SimpleNamespace
    env_info: SimpleNamespace
    flat_obs_dim: int


@pytest.fixture
def fake_state() -> FakeState:
    rollout_steps = 8
    num_envs = 2
    obs_dim = 4

    buffer_config = SimpleNamespace(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        mini_batch_envs=num_envs,
        lstm_hidden_size=4,
    )

    cfg = SimpleNamespace(buffer_config=buffer_config)

    env_info = SimpleNamespace(
        action_dim=1,
        obs_dim=obs_dim,
    )

    return FakeState(
        cfg=cfg,
        env_info=env_info,
        flat_obs_dim=obs_dim,
    )
