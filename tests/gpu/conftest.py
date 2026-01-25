import pytest
import torch
from lstmppo.trainer import LSTMPPOTrainer

@pytest.fixture
def deterministic_trainer():
    trainer = LSTMPPOTrainer.for_validation()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.policy.eval()
    trainer.state.cfg.trainer.debug_mode = True

    return trainer