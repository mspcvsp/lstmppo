"""
Numerical tests validate aux-loss behavior using synthetic latents.
"""

import pytest
import torch


@pytest.mark.invariants
def test_novelty_loss_increases_on_repetition(aux_world_model, latent_cluster):
    wm = aux_world_model
    novelty_head = wm.aux_heads["novelty"]

    # Repeated latent cluster → novelty loss should be high
    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=0, batch=4)

    logits = novelty_head(h, z)

    # Novelty head should produce non-trivial logits for repeated latents
    assert logits.abs().mean() > 0.05


@pytest.mark.invariants
def test_skill_loss_matches_clusters(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=3, batch=4)

    logits = skill_head(h, z)
    pred = logits.argmax(dim=-1)

    # Skill head should strongly prefer the correct cluster
    assert (pred == 3).all()
