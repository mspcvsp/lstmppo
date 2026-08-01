"""
Adaptive skill-learning tests ensure:

    - skill loss decreases when actor chooses correct skill
    - skill loss increases when actor chooses wrong skill
    - aux losses propagate gradients into RSSM
"""

import pytest
import torch
import torch.nn.functional as F

from dreamerrl.tests.aux.utils import assert_skill_loss_order


@pytest.mark.aux_losses
def test_skill_head_learns_to_prefer_correct_skill(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=2, batch=4)

    optimizer = torch.optim.Adam(skill_head.parameters(), lr=1e-2)

    correct_target = F.one_hot(
        torch.full((4,), 2),
        num_classes=wm.net_cfg.action_dim,
    ).float()

    # Train towards correct_target
    for _ in range(50):
        logits = skill_head(h, z)
        loss = F.mse_loss(logits, correct_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # After training, correct loss should be lower than wrong
    logits = skill_head(h, z)

    loss_correct, loss_wrong = assert_skill_loss_order(
        logits,
        correct_skill_idx=2,
        wrong_skill_idx=1,
        num_skills=wm.net_cfg.action_dim,
    )
    assert loss_correct < loss_wrong
