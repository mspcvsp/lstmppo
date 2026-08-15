import numpy as np
import torch

from dreamerrl.env.minigrid.minigrid_parallel_env import MinigridParallelEnv
from dreamerrl.env.minigrid.minigrid_wrappers import MinigridVecEnv


def _rollout(env, num_steps: int = 10):
    traj = []
    out = env.reset()
    traj.append(out)

    for _ in range(num_steps):
        batch_size = out["state"].shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long)
        out = env.step(actions)
        traj.append(out)

    return traj


def test_parallel_env_reset_deterministic(minigrid_env_cfg, device):
    env1 = MinigridParallelEnv(minigrid_env_cfg, device=device)
    env2 = MinigridParallelEnv(minigrid_env_cfg, device=device)

    out1 = env1.reset(seed=minigrid_env_cfg.seed)
    out2 = env2.reset(seed=minigrid_env_cfg.seed)

    s1 = out1["state"].detach().cpu().numpy()
    s2 = out2["state"].detach().cpu().numpy()

    assert np.allclose(s1, s2), "Parallel Minigrid reset must be deterministic under same seed."


def test_parallel_env_step_deterministic(minigrid_env_cfg, device):
    env1 = MinigridParallelEnv(minigrid_env_cfg, device=device)
    env2 = MinigridParallelEnv(minigrid_env_cfg, device=device)

    traj1 = _rollout(env1, num_steps=15)
    traj2 = _rollout(env2, num_steps=15)

    for t1, t2 in zip(traj1, traj2):
        s1 = t1["state"].detach().cpu().numpy()
        s2 = t2["state"].detach().cpu().numpy()
        r1 = t1["reward"].detach().cpu().numpy()
        r2 = t2["reward"].detach().cpu().numpy()
        assert np.allclose(s1, s2), "States must match across deterministic runs."
        assert np.allclose(r1, r2), "Rewards must match across deterministic runs."


def test_vec_env_batch_size_and_flags(minigrid_env_cfg, device):
    env = MinigridVecEnv(minigrid_env_cfg, device=device)
    out = env.reset()

    batch_size = env.batch_size
    assert out["state"].shape[0] == batch_size
    assert out["reward"].shape[0] == batch_size
    assert out["is_first"].shape[0] == batch_size
    assert out["is_last"].shape[0] == batch_size
    assert out["is_terminal"].shape[0] == batch_size

    # After one step, is_first should be False for all envs
    actions = torch.zeros(batch_size, dtype=torch.long)
    out2 = env.step(actions)

    assert torch.all(out2["is_first"] == torch.zeros_like(out2["is_first"])), (
        "is_first must be False after the first step."
    )


def test_vec_env_terminal_flags_consistency(minigrid_env_cfg, device):
    env = MinigridVecEnv(minigrid_env_cfg, device=device)
    out = env.reset()

    batch_size = env.batch_size
    actions = torch.zeros(batch_size, dtype=torch.long)

    # Roll until at least one env terminates/truncates
    for _ in range(minigrid_env_cfg.max_episode_steps + 5):
        out = env.step(actions)
        is_last = out["is_last"]
        is_terminal = out["is_terminal"]

        # Contract: whenever is_terminal is True, is_last must also be True
        assert torch.all(is_terminal <= is_last), "is_terminal implies is_last for MinigridVecEnv."

        if bool(is_last.any()):
            break
