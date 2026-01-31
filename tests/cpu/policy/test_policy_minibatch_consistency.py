import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyEvalInput, PolicyInput

pytestmark = pytest.mark.policy


def test_policy_minibatch_consistency(trainer_state: TrainerState):
    """
    Smoke test: step-mode and sequence-mode evaluation must produce
    identical values and log-probs when given the same inputs and
    initial hidden state.
    """

    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    trainer_state.cfg.lstm.enc_hidden_size = 16
    trainer_state.cfg.lstm.lstm_hidden_size = 8

    policy = LSTMPPOPolicy(trainer_state)

    B = 4
    T = 5
    H = trainer_state.cfg.lstm.lstm_hidden_size
    # Random observations and actions
    obs = torch.randn(B, T, trainer_state.env_info.flat_obs_dim)
    actions = torch.randint(0, trainer_state.env_info.action_dim, (B, T))

    # Initial hidden state
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    # --- Step-mode unroll ---
    h_step, c_step = h0.clone(), c0.clone()
    logps_step = []
    values_step = []

    for t in range(T):
        out = policy.forward(PolicyInput(obs=obs[:, t], hxs=h_step, cxs=c_step))
        logps_step.append(policy.evaluate_actions(out, actions[:, t]).logprobs)
        values_step.append(out.values)
        h_step, c_step = out.new_hxs, out.new_cxs

    logps_step = torch.stack(logps_step, dim=1)  # (B, T)
    values_step = torch.stack(values_step, dim=1)  # (B, T)

    # --- Sequence-mode evaluation ---
    out_seq = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs.transpose(0, 1),  # (T, B, F)
            actions=actions.transpose(0, 1),  # (T, B)
            hxs=h0,
            cxs=c0,
        )
    )

    logps_seq = out_seq.logprobs.transpose(0, 1)  # (B, T)
    values_seq = out_seq.values.transpose(0, 1)  # (B, T)

    # --- Consistency checks ---
    assert torch.allclose(logps_step, logps_seq)
    assert torch.allclose(values_step, values_seq)
