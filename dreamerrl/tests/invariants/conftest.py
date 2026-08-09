import pytest

from dreamerrl.env.crafter.crafter_env_wrapper import CrafterEnvWrapper


@pytest.fixture
def crafter_cfg():
    return type(
        "Cfg",
        (),
        {
            "env_id": "Crafter-v1",
            "seed": 0,
            "max_episode_steps": 10000,
        },
    )


@pytest.fixture
def crafter_env(crafter_cfg):
    return CrafterEnvWrapper(crafter_cfg)


@pytest.fixture
def crafter_cfg_short():
    return type(
        "Cfg",
        (),
        {
            "env_id": "Crafter-v1",
            "seed": 0,
            "max_episode_steps": 5,
        },
    )


@pytest.fixture
def crafter_env_short(crafter_cfg_short):
    return CrafterEnvWrapper(crafter_cfg_short)
