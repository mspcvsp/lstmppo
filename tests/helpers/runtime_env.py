import numpy as np
from gymnasium.spaces import Box

from lstmppo.runtime_env_info import RuntimeEnvInfo


def make_runtime_env_info(
    obs_dim: int = 4,
    action_dim: int = 3,
):
    """
    Returns a RuntimeEnvInfo suitable for unit tests.
    Uses a real gymnasium Box space to satisfy Pylance.
    """

    obs_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32,
    )

    # Pylance-safe narrowing
    assert isinstance(action_dim, int)

    return RuntimeEnvInfo(
        obs_space=obs_space,
        flat_obs_dim=obs_dim,
        action_dim=action_dim,
    )
