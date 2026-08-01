import torch
import torch.nn.functional as F


def assert_skill_loss_order(logits, correct_skill_idx, wrong_skill_idx, num_skills):
    """
    Compares MSE skill losses for correct vs wrong skill targets.

    - logits: (B, num_skills)
    - correct_skill_idx: int
    - wrong_skill_idx: int
    - num_skills: wm.net.action_dim

    This removes hard-coded values from tests and ensures consistency.
    """

    target_correct = F.one_hot(
        torch.full((logits.size(0),), correct_skill_idx),
        num_classes=num_skills
    ).float()

    target_wrong = F.one_hot(
        torch.full((logits.size(0),), wrong_skill_idx),
        num_classes=num_skills
    ).float()

    loss_correct = F.mse_loss(logits, target_correct)
    loss_wrong = F.mse_loss(logits, target_wrong)

    return loss_correct, loss_wrong
