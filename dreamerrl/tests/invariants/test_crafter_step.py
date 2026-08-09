import pytest
import torch


@pytest.mark.crafter_env_invariants
def test_step_deterministic(crafter_env):
    crafter_env.reset()
    seq1 = []
    for _ in range(5):
        seq1.append(crafter_env.step(torch.tensor([0]))["state"])

    crafter_env.reset()
    seq2 = []
    for _ in range(5):
        seq2.append(crafter_env.step(torch.tensor([0]))["state"])

    for a, b in zip(seq1, seq2):
        assert torch.allclose(a, b)


@pytest.mark.crafter_env_invariants
def test_boundary_flags(crafter_env):
    out = crafter_env.reset()
    assert out["is_first"].item() is True
    assert out["is_last"].item() is False
    assert out["is_terminal"].item() is False

    done = False
    while not done:
        out = crafter_env.step(torch.tensor([0]))
        done = out["is_last"].item()

    assert out["is_last"].item() is True
    assert out["is_terminal"].item() in (True, False)


@pytest.mark.crafter_env_invariants
def test_action_validity(crafter_env):
    assert crafter_env.action_dim == 17

    out = crafter_env.step(torch.tensor([0]))
    assert out["reward"].shape == (1,)


@pytest.mark.crafter_env_invariants
def test_max_episode_steps_truncation(crafter_env_short):
    crafter_env_short.reset()

    for _ in range(5):
        out = crafter_env_short.step(torch.tensor([0]))

    assert out["is_last"].item() is True
