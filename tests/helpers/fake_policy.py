"""
Factory for constructing a valid LSTMPPOPolicy using the FakeState
defined in tests/helpers/fake_state.py.

This keeps test setup clean and ensures the policy is always created
with a structurally correct TrainerStateProtocol.
"""

from lstmppo.policy import LSTMPPOPolicy
from tests.helpers.fake_state import TrainerStateProtocol, make_fake_state


def make_fake_policy(
    *,
    rollout_steps: int = 8,
    num_envs: int = 2,
    obs_dim: int = 4,
    hidden_size: int = 4,
) -> LSTMPPOPolicy:
    state: TrainerStateProtocol = make_fake_state(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        hidden_size=hidden_size,
    )

    return LSTMPPOPolicy(state)  # type: ignore[arg-type]
