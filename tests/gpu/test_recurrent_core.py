import torch
import pytest
from lstmppo.trainer import LSTMPPOTrainer

pytestmark = pytest.mark.gpu


def test_tbptt_determinism(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    # This will assert internally
    trainer.validate_tbptt()


def test_rollout_replay_determinism(deterministic_trainer):

    trainer = deterministic_trainer

    trainer.validate_lstm_state_flow()  # asserts internally


def test_hidden_state_alignment(deterministic_trainer):
    
    trainer = deterministic_trainer
    trainer.collect_rollout()
    batch = next(trainer.buffer.get_recurrent_minibatches())

    assert batch.hxs.shape[0] == trainer.rollout_steps
    assert batch.hxs.shape[1] == trainer.num_envs

    assert batch.cxs.shape[0] == trainer.rollout_steps
    assert batch.cxs.shape[1] == trainer.num_envs

    trainer.validate_lstm_state_flow()


def test_mask_correctness(deterministic_trainer):

    trainer = deterministic_trainer
    trainer.collect_rollout()

    buf = trainer.buffer

    assert buf.mask.shape == buf.terminated.shape

    expected = 1.0 - (buf.terminated | buf.truncated).float()
    assert torch.allclose(buf.mask, expected)

    assert torch.all(buf.mask[buf.terminated] == 0)
    assert torch.all(buf.mask[buf.truncated] == 0)

    alive = ~(buf.terminated | buf.truncated)
    assert torch.all(buf.mask[alive] == 1)
