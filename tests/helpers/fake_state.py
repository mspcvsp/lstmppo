"""
Reusable FakeTrainerState for tests.

This helper encodes the minimal set of invariants required by:

- LSTMPPOPolicy.__init__
- RecurrentRolloutBuffer
- evaluate_actions_sequence
- TBPTT chunking
- Aux prediction heads (pred_obs, pred_raw)

It ensures:
- obs_dim, flat_obs_dim, and env_info.obs_dim are consistent
- cfg.lstm subtree contains all attributes accessed by the policy
- cfg.trainer.debug_mode exists
- env_info.obs_space is non-None so obs_dim != 0
- a real nn.LSTM module is provided for state.lstm
"""

from dataclasses import dataclass
from types import SimpleNamespace
import torch.nn as nn


@dataclass
class FakeState:
    cfg: SimpleNamespace
    env_info: SimpleNamespace
    flat_obs_dim: int
    lstm: nn.Module


def make_fake_state(
    rollout_steps: int = 8,
    num_envs: int = 2,
    obs_dim: int = 4,
    hidden_size: int = 4,
) -> FakeState:
    # --- Buffer config ---
    buffer_config = SimpleNamespace(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        mini_batch_envs=num_envs,
        lstm_hidden_size=hidden_size,
    )

    # --- LSTM config subtree (policy expects all of these) ---
    lstm_cfg = SimpleNamespace(
        lstm_ar_coef=0.0,
        lstm_tar_coef=0.0,
        lstm_dropout=0.0,
        lstm_layer_norm=False,
        enc_hidden_size=128,
        lstm_hidden_size=hidden_size,
        dropconnect_p=0.0,
    )

    # --- Trainer config subtree ---
    trainer_cfg = SimpleNamespace(
        debug_mode=False,
    )

    cfg = SimpleNamespace(
        buffer_config=buffer_config,
        lstm=lstm_cfg,
        trainer=trainer_cfg,
    )

    # --- Env info ---
    dummy_obs_space = SimpleNamespace(shape=(obs_dim,))
    env_info = SimpleNamespace(
        action_dim=1,
        obs_dim=obs_dim,
        obs_space=dummy_obs_space,  # ensures obs_dim != 0
        flat_obs_dim=obs_dim,
    )

    # --- Real LSTM module ---
    lstm = nn.LSTM(
        input_size=obs_dim,
        hidden_size=hidden_size,
        batch_first=True,
    )

    return FakeState(
        cfg=cfg,
        env_info=env_info,
        flat_obs_dim=obs_dim,
        lstm=lstm,
    )
