import pytest
import torch


@pytest.mark.crafter_env_invariants
def test_obs_shape_and_dtype(crafter_env):
    out = crafter_env.reset()
    state = out["state"]

    assert state.shape == (1, 64, 64, 3)
    assert state.dtype == torch.float32


@pytest.mark.crafter_env_invariants
def test_obs_normalization(crafter_env):
    out = crafter_env.reset()
    state = out["state"]

    assert torch.all(state >= 0.0)
    assert torch.all(state <= 1.0)


@pytest.mark.crafter_env_invariants
def test_reset_deterministic(crafter_env):
    s1 = crafter_env.reset()["state"]
    s2 = crafter_env.reset()["state"]

    assert torch.allclose(s1, s2)


@pytest.mark.crafter_env_invariants
def test_batch_dimension(crafter_env):
    out = crafter_env.reset()
    for key in ["state", "reward", "is_first", "is_last", "is_terminal"]:
        assert out[key].shape[0] == 1
