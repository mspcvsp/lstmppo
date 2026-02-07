import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import PolicyEvalInput
from tests.cpu.aux_pred.conftest import FakeState


def test_aux_gradients(fake_state: FakeState):
    policy = LSTMPPOPolicy(fake_state)  # type: ignore[arg-type]

    policy.train()

    T, B, obs_dim = 4, 1, fake_state.flat_obs_dim
    obs = torch.randn(T, B, obs_dim)
    h0 = torch.zeros(B, policy.lstm.hidden_size)
    c0 = torch.zeros(B, policy.lstm.hidden_size)
    actions = torch.zeros(T, B, dtype=torch.long)

    out = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions))

    # pred_obs and pred_raw must require grad
    assert out.pred_obs is not None
    assert out.pred_raw is not None

    assert out.pred_obs.requires_grad
    assert out.pred_raw.requires_grad

    # targets must NOT require grad
    next_obs = obs.roll(-1, dims=0)
    next_obs[-1] = 0
    assert not next_obs.requires_grad

    # simple aux loss
    loss = ((out.pred_obs - next_obs) ** 2).mean()
    loss.backward()

    # hidden states must NOT have gradients
    assert out.new_hxs.grad is None
    assert out.new_cxs.grad is None
