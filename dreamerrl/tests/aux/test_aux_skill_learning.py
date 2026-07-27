"""
Adaptive skill-learning tests ensure:

    - skill loss decreases when actor chooses correct skill
    - skill loss increases when actor chooses wrong skill
    - aux losses propagate gradients into RSSM
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.invariants
def test_skill_loss_decreases_with_correct_actor_choice(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=2, batch=4)

    logits = skill_head(h, z)

    loss_correct = F.cross_entropy(logits, torch.full((4,), 2))
    loss_wrong = F.cross_entropy(logits, torch.full((4,), 1))

    assert loss_correct < loss_wrong


@pytest.mark.invariants
def test_skill_loss_increases_with_wrong_actor_choice(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=1, batch=4)

    logits = skill_head(h, z)

    loss_correct = F.cross_entropy(logits, torch.full((4,), 1))
    loss_wrong = F.cross_entropy(logits, torch.full((4,), 0))

    assert loss_wrong > loss_correct


@pytest.mark.invariants
def test_aux_loss_propagates_gradients_into_rssm(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    # h must require grad to test gradient flow
    h = torch.randn(4, wm.latent.deter_size, requires_grad=True)
    z = latent_cluster(cluster_id=0, batch=4)

    logits = skill_head(h, z)
    loss = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long))

    grads = torch.autograd.grad(loss, h, retain_graph=True)[0]

    assert grads.abs().sum() > 0
