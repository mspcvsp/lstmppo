import pytest
import torch

from dreamerrl.env.minigrid.minigrid_parallel_env import MinigridParallelEnv
from dreamerrl.env.minigrid.minigrid_wrappers import MinigridVecEnv
from dreamerrl.utils.types import EnvironmentConfig


@pytest.fixture(scope="session")
def device():
    """
    Use CPU for invariants tests.
    GPU determinism is tested separately in functional/invariants.
    """
    return torch.device("cpu")


@pytest.fixture(scope="session")
def minigrid_env_cfg():
    """
    Shared deterministic config for Minigrid invariants tests.
    Mirrors PopGym invariants config structure.
    """
    return EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        num_envs=4,
        seed=123,
        max_episode_steps=50,
        deterministic=True,
    )


@pytest.fixture
def minigrid_parallel_env(minigrid_env_cfg, device):
    """
    Parallel Minigrid environment (AsyncVectorEnv).
    Used for:
        - reset invariants
        - step invariants
        - batch invariants
        - timelimit invariants
    """
    env = MinigridParallelEnv(minigrid_env_cfg, device=device)
    env.reset(seed=minigrid_env_cfg.seed)
    return env


@pytest.fixture
def minigrid_vec_env(minigrid_env_cfg, device):
    """
    SyncVectorEnv-based Minigrid wrapper.
    Used for:
        - flattening invariants
        - state contract tests
        - flag consistency tests
    """
    env = MinigridVecEnv(minigrid_env_cfg, device=device)
    env.reset(seed=minigrid_env_cfg.seed)
    return env
