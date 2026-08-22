import pytest
import torch


@pytest.mark.invariants
@pytest.mark.minihack_invariants
def test_minihack_contract(minihack_env):
    out = minihack_env.reset()

    # Core keys
    assert "state" in out
    assert "prev_action" in out
    assert "reward" in out
    assert "is_first" in out
    assert "is_last" in out
    assert "is_terminal" in out
    assert "info" in out

    state = out["state"]
    prev_action = out["prev_action"]

    # State shape + dtype
    assert state.shape == (minihack_env.batch_size, minihack_env.obs_dim)
    assert state.dtype == torch.float32

    # Prev action shape + dtype
    assert prev_action.shape == (minihack_env.batch_size, minihack_env.action_dim)
    assert prev_action.dtype == torch.float32

    # Action/obs dims positive
    assert minihack_env.action_dim > 0
    assert minihack_env.obs_dim > 0
