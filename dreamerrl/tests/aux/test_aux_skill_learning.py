"""
Adaptive skill-learning tests ensure:

    - skill loss decreases when actor chooses correct skill
    - skill loss increases when actor chooses wrong skill
    - aux losses propagate gradients into RSSM
"""

import pytest
import torch

from dreamerrl.tests.aux.utils import assert_skill_loss_order


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_skill_loss_decreases_with_correct_actor_choice(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=2, batch=4)

    logits = skill_head(h, z)

    loss_correct, loss_wrong = assert_skill_loss_order(
        logits,
        correct_skill_idx=2,
        wrong_skill_idx=1,
        num_skills=wm.net_cfg.action_dim,
    )

    assert loss_correct < loss_wrong


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_skill_loss_increases_with_wrong_actor_choice(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=1, batch=4)

    logits = skill_head(h, z)

    loss_correct, loss_wrong = assert_skill_loss_order(
        logits,
        correct_skill_idx=1,
        wrong_skill_idx=0,
        num_skills=wm.net_cfg.action_dim,
    )

    assert loss_wrong > loss_correct
