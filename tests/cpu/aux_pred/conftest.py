from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@dataclass
class FakeState:
    cfg: SimpleNamespace
    flat_obs_dim: int


@pytest.fixture
def fake_state() -> FakeState:
    cfg = SimpleNamespace()
    cfg.buffer_config = SimpleNamespace(
        rollout_steps=5,
        num_envs=2,
        lstm_hidden_size=4,
    )

    return FakeState(
        cfg=cfg,
        flat_obs_dim=3,
    )
