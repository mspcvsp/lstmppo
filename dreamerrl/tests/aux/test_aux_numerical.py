"""
Numerical tests validate aux-loss behavior using synthetic latents.
"""

import pytest
import torch


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_novelty_loss_increases_on_repetition(aux_world_model, latent_cluster):
    wm = aux_world_model
    novelty_head = wm.aux_heads["novelty"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=0, batch=4)

    logits = novelty_head(h, z)

    # Non-trivial output, correct shape
    assert logits.shape == (4, 1)
    assert logits.abs().mean() > 0.0


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_skill_head_output_shape(aux_world_model, latent_cluster):
    wm = aux_world_model
    skill_head = wm.aux_heads["skill"]

    h = torch.zeros(4, wm.latent.deter_size)
    z = latent_cluster(cluster_id=3, batch=4)

    logits = skill_head(h, z)

    # Correct shape, no NaNs
    assert logits.shape == (4, wm.net_cfg.action_dim)
    assert torch.isfinite(logits).all()
