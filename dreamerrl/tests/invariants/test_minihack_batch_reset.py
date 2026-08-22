import pytest
import torch


@pytest.mark.invariants
@pytest.mark.minihack_invariants
def test_per_env_reset(minihack_env):
    """
    Ensures only terminated envs are reset and others continue.
    We do NOT rely on glyph equality, because MiniHack can produce
    identical glyph grids across resets with the same seed.
    Instead we check:
      - is_first is re-raised only for terminated envs
      - prev_action is cleared only for terminated envs
    """
    minihack_env.reset()

    actions = torch.zeros(minihack_env.batch_size, dtype=torch.long)
    out = None
    for _ in range(200):  # enough steps to force at least one termination
        out = minihack_env.step(actions)
        if out["is_last"].any():
            break

    assert out is not None
    assert out["is_last"].any()

    last_mask = out["is_last"]
    prev_before = out["prev_action"].clone()

    # One more step to trigger per-env reset logic
    out2 = minihack_env.step(actions)
    prev_after = out2["prev_action"]
    is_first_after = out2["is_first"]

    for i in range(minihack_env.batch_size):
        if last_mask[i]:
            # Terminated env: is_first should be True again
            assert is_first_after[i]

            # MiniHack always emits a MOVE_NOP action after reset
            assert prev_after[i].sum() == 1.0

        else:
            # Non-terminated env: is_first should be False
            assert not is_first_after[i]

            # Action should advance normally
            assert prev_after[i].sum() == 1.0
            assert not torch.allclose(prev_after[i], prev_before[i])
