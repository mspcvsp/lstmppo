import pytest

from lstmppo.trainer_state import TrainerState
from lstmppo.types import Config
from tests.helpers.fake_state import TrainerStateProtocol, make_fake_state
from tests.helpers.runtime_env import make_runtime_env_info


@pytest.fixture
def trainer_state():
    cfg = Config()
    cfg.trainer.debug_mode = True

    state = TrainerState(cfg)
    state.env_info = make_runtime_env_info()
    return state


@pytest.fixture
def fake_state() -> TrainerStateProtocol:
    return make_fake_state()
