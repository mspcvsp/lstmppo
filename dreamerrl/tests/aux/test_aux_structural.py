"""
Structural tests ensure aux heads exist, aux logits appear when enabled, and aux logits disappear when
disable_aux_losses=True.
"""

import pytest
import torch


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_aux_heads_instantiation(aux_world_model):
    wm = aux_world_model
    assert "novelty" in wm.aux_heads
    assert "skill" in wm.aux_heads
    assert isinstance(wm.aux_heads["novelty"], torch.nn.Module)
    assert isinstance(wm.aux_heads["skill"], torch.nn.Module)


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_aux_logits_present(aux_world_model):
    wm = aux_world_model

    prev = wm.init_state(batch_size=4)
    obs = torch.randn(4, wm.flat_obs_dim)
    action = torch.zeros(4, wm.net_cfg.action_dim)

    out = wm.observe_step(prev_state=prev, obs=obs, action=action)

    assert "aux_logits" in out
    assert "novelty" in out["aux_logits"]
    assert "skill" in out["aux_logits"]


@pytest.mark.invariants
@pytest.mark.aux_losses
def test_aux_logits_disabled(world_model_aux_losses_disabled):
    wm = world_model_aux_losses_disabled

    prev = wm.init_state(batch_size=4)
    obs = torch.randn(4, wm.flat_obs_dim)
    action = torch.zeros(4, wm.net_cfg.action_dim)

    out = wm.observe_step(prev_state=prev, obs=obs, action=action)

    assert out["aux_logits"] == {}
