# dreamerrl/env/env_factory.py

from __future__ import annotations

from dreamerrl.env.minigrid.minigrid_parallel_env import MinigridParallelEnv
from dreamerrl.env.minigrid.minigrid_wrappers import MinigridVecEnv
from dreamerrl.env.minihack.minihack_parallel_env import MiniHackParallelEnv
from dreamerrl.env.minihack.minihack_wrappers import MiniHackVecEnv
from dreamerrl.env.popgym.popgym_parallel_env import PopGymParallelEnv
from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv


def make_env(cfg, device, probe=None):
    """
    Unified Dreamer-V3 environment factory.

    Selects the correct environment wrapper based on env_id prefix:
        popgym-*
        minigrid-*
        minihack-*
        cage2-*

    All wrappers implement the Dreamer-native EnvInterface:
        reset() -> dict(state, reward, is_first, is_last, is_terminal, prev_action)
        step(action) -> same dict
        batch_size, obs_dim, action_dim
    """

    env_id = cfg.env.env_id.lower()

    # ---------------------------------------------------------
    # PopGym
    # ---------------------------------------------------------
    if env_id.startswith("popgym"):
        if cfg.env.parallel:
            return PopGymParallelEnv(cfg.env, device=device)
        return PopGymVecEnv(cfg.env, device=device, probe=probe)

    # ---------------------------------------------------------
    # Minigrid
    # ---------------------------------------------------------
    if env_id.startswith("minigrid"):
        if cfg.env.parallel:
            return MinigridParallelEnv(cfg.env, device=device)
        return MinigridVecEnv(cfg.env, device=device)

    # ---------------------------------------------------------
    # MiniHack
    # ---------------------------------------------------------
    if env_id.startswith("minihack"):
        if cfg.env.parallel:
            return MiniHackParallelEnv(cfg.env, device=device)
        return MiniHackVecEnv(cfg.env, device=device)

    raise ValueError(f"Unknown environment family for env_id={cfg.env.env_id}")
