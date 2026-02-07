import pytest
import torch

from tests.helpers.fake_policy import make_fake_policy
from tests.helpers.fake_state import TrainerStateProtocol
from lstmppo.types import PolicyInput

pytestmark = pytest.mark.drift


def test_cell_state_drift_accumulates_over_time(fake_state: TrainerStateProtocol):
    policy = make_fake_policy()
    policy.eval()

    B = 3
    H = fake_state.cfg.lstm.lstm_hidden_size
    D = fake_state.flat_obs_dim

    lengths = [5, 50]
    num_samples = 20

    avg_drifts = []

    for L in lengths:
        drifts = []
        for _ in range(num_samples):
            obs = torch.randn(B, L, D)
            h0 = torch.zeros(B, H)
            c0 = torch.zeros(B, H)

            out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

            drifts.append(out.gates.c_gates.pow(2).mean())

        avg_drifts.append(torch.stack(drifts).mean())

    eps = 1e-5
    assert avg_drifts[1] + eps >= avg_drifts[0] - eps
