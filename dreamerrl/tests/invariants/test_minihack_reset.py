import torch


def test_reset_deterministic(minihack_env):
    out1 = minihack_env.reset()
    out2 = minihack_env.reset()

    # Shape determinism
    assert out1["state"].shape == out2["state"].shape

    # Reward determinism
    assert torch.allclose(out1["reward"], out2["reward"])

    # First-step flags
    assert torch.all(out1["is_first"])
    assert torch.all(out2["is_first"])
