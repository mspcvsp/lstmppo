import torch
import pytest
from lstmppo.trainer import LSTMPPOTrainer

# ------------------------------------------------------------
# Deterministic trainer fixture
# ------------------------------------------------------------
@pytest.fixture
def deterministic_trainer():
    trainer = LSTMPPOTrainer.for_validation()

    # Hard determinism
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.policy.eval()
    trainer.state.cfg.trainer.debug_mode = True

    return trainer


# ------------------------------------------------------------
# 1. TBPTT determinism test
# ------------------------------------------------------------
def test_tbptt_determinism(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    # This will assert internally
    trainer.validate_tbptt()


# ------------------------------------------------------------
# 2. Rollout → Replay determinism test
# ------------------------------------------------------------
def test_rollout_replay_determinism(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.validate_lstm_state_flow()  # asserts internally


# ------------------------------------------------------------
# 3. Hidden-state alignment test
# ------------------------------------------------------------
def test_hidden_state_alignment(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    batch = next(trainer.buffer.get_recurrent_minibatches())

    # Hidden states must be (T, B, H)
    assert batch.hxs.shape[0] == trainer.rollout_steps
    assert batch.hxs.shape[1] == trainer.num_envs

    # 2. Hidden states must be PRE-STEP states
    #    So hxs[t] must equal the state used to compute value/logprob at t.
    #    We validate this by ensuring the state-flow validator passes.
    trainer.validate_lstm_state_flow()

