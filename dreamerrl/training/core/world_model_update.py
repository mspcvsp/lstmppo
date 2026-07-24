from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.types import WorldModelMetrics


def world_model_training_step(
    world_model,
    batch: Dict[str, torch.Tensor],
    kl_scale: float = 1.0,
) -> WorldModelMetrics:
    device = next(world_model.parameters()).device

    # -------------------------------------------------------------
    # Extract batch
    # -------------------------------------------------------------
    if "obs" in batch:
        obs = batch["obs"].to(device)
    elif "state" in batch:
        obs = batch["state"].to(device)
    else:
        raise KeyError("Batch must contain 'obs' or 'state'")

    B, L, _ = obs.shape

    reward = batch["reward"].to(device)

    # -------------------------------------------------------------
    # Short-horizon return target (auxiliary reward)
    # r_sh[t] = r[t] + γ r[t+1] + γ^2 r[t+2]
    # -------------------------------------------------------------
    gamma = world_model.net_cfg.discount

    if L >= 3:
        short_horizon = reward[:, :-2] + gamma * reward[:, 1:-1] + (gamma**2) * reward[:, 2:]
        # pad last 2 steps with zeros
        pad = torch.zeros(B, 2, device=device)
        short_horizon = torch.cat([short_horizon, pad], dim=1)
    else:
        # fallback for tiny sequences (tests)
        short_horizon = reward.clone()

    # continuation target = 1.0 if episode continues, 0.0 if terminal
    if "is_terminal" in batch:
        cont_target = 1.0 - batch["is_terminal"].to(device)
    else:
        # fallback for tests that don't include is_terminal
        cont_target = torch.ones_like(reward)

    # -------------------------------------------------------------
    # Roll out RSSM over sequence
    # -------------------------------------------------------------
    state = world_model.init_state(B)
    posts = []
    priors = []

    for t in range(L):
        action_t = batch["action"][:, t]  # (B,)

        # Convert one-hot → discrete ID if needed
        if action_t.dtype != torch.long:
            action_t = action_t.argmax(dim=-1)

        action_one_hot = F.one_hot(action_t, num_classes=world_model.net_cfg.action_dim).float()  # (B, action_dim)

        out = world_model.observe_step(
            prev_state=state,
            obs=obs[:, t],
            action=action_one_hot,
            reward=batch["reward"][:, t],
        )

        state = out["post"]
        posts.append(out["post"].post_stats)
        priors.append(out["prior"].prior_stats)

    # (B, L, deter), (B, L, K, C)
    h = torch.stack([s["h"] for s in posts], dim=1)
    z = torch.stack([s["z"] for s in posts], dim=1)

    # -------------------------------------------------------------
    # Flatten for heads
    # -------------------------------------------------------------
    z_factored = z.reshape(
        B * L,
        world_model.latent.num_classes,
        world_model.latent.stoch_size,
    )
    h_flat = h.reshape(B * L, -1)

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------
    recon = world_model.decoder(h_flat, z_factored).reshape(B, L, -1)

    reward_main_logits, _ = world_model.reward_heads(h_flat, z_factored)
    reward_main_logits = reward_main_logits.reshape(B, L, world_model.net_cfg.value_bins)

    aux_logits = {name: head(h_flat, z_factored) for name, head in world_model.aux_heads.items()}
    cont_logits = world_model.continue_head(h_flat, z_factored).reshape(B, L, world_model.net_cfg.value_bins)

    # -------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------
    recon_target = symlog(obs)
    recon_loss = F.mse_loss(recon, recon_target)

    reward_loss = world_model.reward_heads.main.loss_from_logits(reward_main_logits, reward)
    cont_loss = world_model.continue_head.loss_from_logits(cont_logits, cont_target)
    L_pred = recon_loss + reward_loss + cont_loss

    aux_losses = []

    # Skip aux losses in reproducibility mode
    if not getattr(world_model.net_cfg, "disable_aux_losses", False):
        for cfg in world_model.aux_objectives:
            name = cfg.name
            logits = aux_logits[name].reshape(B, L, -1)  # (B, L, dim)
            target = cfg.fn(batch, gamma)  # (B, L, dim)

            if target.dim() == 2:
                target = target.unsqueeze(-1)  # ensures (B, L, 1)

            # normalize target
            target = (target - target.mean()) / (target.std() + 1e-6)

            head = world_model.aux_heads[name]
            loss = head.loss_from_logits(logits, target)

            # clamp loss to prevent latent drift
            loss = torch.clamp(loss, -1.0, 1.0)  # >>> NEW

            aux_losses.append(loss)

    if aux_losses:
        aux_loss = sum(aux_losses) / len(aux_losses)
        L_pred = L_pred + world_model.net_cfg.aux_reward_scale * aux_loss

    # -------------------------------------------------------------
    # KL losses (already computed in observe_step)
    # -------------------------------------------------------------
    kl_dyn = torch.stack([p["kl_dyn"] for p in posts], dim=1).mean()
    kl_rep = torch.stack([p["kl_rep"] for p in posts], dim=1).mean()

    total_loss = L_pred + kl_scale * (kl_dyn + kl_rep)

    return WorldModelMetrics(
        total_loss=total_loss,
        recon_loss=recon_loss,
        reward_loss=reward_loss,
        cont_loss=cont_loss,
        kl_dyn=kl_dyn,
        kl_rep=kl_rep,
        aux_losses=aux_losses,
    )
