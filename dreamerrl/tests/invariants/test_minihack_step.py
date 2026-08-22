import pytest
import torch


@pytest.mark.invariants
@pytest.mark.minihack_invariants
def test_minihack_step(minihack_env):
    minihack_env.reset()

    actions = torch.zeros(minihack_env.batch_size, dtype=torch.long)
    out = minihack_env.step(actions)

    # Shapes
    assert out["state"].shape == (minihack_env.batch_size, minihack_env.obs_dim)
    assert out["reward"].shape == (minihack_env.batch_size,)
    assert out["is_last"].shape == (minihack_env.batch_size,)
    assert out["is_terminal"].shape == (minihack_env.batch_size,)
    assert out["is_first"].shape == (minihack_env.batch_size,)
    assert out["prev_action"].shape == (minihack_env.batch_size, minihack_env.action_dim)

    # Dtypes
    assert out["state"].dtype == torch.float32
    assert out["reward"].dtype == torch.float32
    assert out["is_last"].dtype == torch.bool
    assert out["is_terminal"].dtype == torch.bool
    assert out["is_first"].dtype == torch.bool
    assert out["prev_action"].dtype == torch.float32

    # prev_action one-hot correctness for action 0
    assert torch.all(out["prev_action"][:, 0] == 1.0)
    assert torch.all(out["prev_action"][:, 1:] == 0.0)
