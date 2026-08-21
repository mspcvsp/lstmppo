import pytest
import torch

from dreamerrl.env.minihack.minihack_wrappers import MiniHackVecEnv
from dreamerrl.utils.types import EnvironmentConfig


@pytest.fixture
def env_cfg():
    return EnvironmentConfig(
        env_id="MiniHack-Room-RedDoor-v0",
        num_envs=4,
        seed=123,
        deterministic=True,
        max_episode_steps=50,
    )


@pytest.fixture
def env(env_cfg):
    return MiniHackVecEnv(env_cfg, device=torch.device("cpu"))


def test_contract(minihack_env):
    out = minihack_env.reset()

    assert "state" in out
    assert "reward" in out
    assert "is_first" in out
    assert "is_last" in out
    assert "is_terminal" in out

    state = out["state"]
    assert state.shape == (minihack_env.batch_size, minihack_env.obs_dim)
    assert state.dtype == torch.float32

    assert minihack_env.action_dim > 0
    assert minihack_env.obs_dim > 0
