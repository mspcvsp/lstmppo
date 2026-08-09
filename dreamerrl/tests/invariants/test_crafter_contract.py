import pytest
import torch


@pytest.mark.crafter_env_invariants
def test_dreamer_contract_keys(crafter_env):
    out = crafter_env.reset()
    required = ["state", "reward", "is_first", "is_last", "is_terminal"]

    for key in required:
        assert key in out


@pytest.mark.crafter_env_invariants
def test_flag_invariants(crafter_env):
    out = crafter_env.reset()
    assert out["is_first"].item() is True
    assert out["is_last"].item() is False
    assert out["is_terminal"].item() is False

    out = crafter_env.step(torch.tensor([0]))
    assert out["is_first"].item() is False  # only true on reset


@pytest.mark.crafter_env_invariants
def test_state_tensor_is_float_and_batched(crafter_env):
    out = crafter_env.reset()
    state = out["state"]

    assert state.dtype == torch.float32
    assert state.ndim == 4
    assert state.shape[0] == 1
