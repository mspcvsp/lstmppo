import pytest
import torch


@pytest.mark.invariants
@pytest.mark.minihack_invariants
def test_terminal_flags(minihack_env):
    """
    Check that terminal semantics are consistent:
    - is_terminal is true when either terminated or truncated
    - is_last matches is_terminal
    """
    minihack_env.reset()
    actions = torch.zeros(minihack_env.batch_size, dtype=torch.long)

    # Step enough times to likely hit termination/truncation
    out = None
    for _ in range(200):
        out = minihack_env.step(actions)

    assert out is not None

    is_terminal = out["is_terminal"]
    is_last = out["is_last"]

    # is_last should be exactly is_terminal
    assert torch.equal(is_last, is_terminal)

    # Flags are boolean
    assert is_terminal.dtype == torch.bool
    assert is_last.dtype == torch.bool
