from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.models.actor import act_in_imagination
from dreamerrl.utils.types import KLConfig, LatentConfig, NetworkConfig

from .aux_objectives import make_aux_heads
from .categorical_kl import structured_kl
from .continue_head import ContinueHead
from .decoder import ObsDecoder
from .multi_reward_head import MultiRewardHead
from .obs_encoder import build_obs_encoder, get_flat_obs_dim
from .posterior import Posterior
from .prior import Prior
from .world_model_core import RSSMCore


@dataclass
class WorldModelState:
    """
    Dreamer‑V3 RSSM state:

        h_t: deterministic state (B, deter_size)
        z_t: factored discrete latent (B, K, C)
    """

    h: torch.Tensor
    z: torch.Tensor
    prior_stats: Optional[Dict[str, torch.Tensor]] = None
    post_stats: Optional[Dict[str, torch.Tensor]] = None

    def to(self, device: torch.device) -> "WorldModelState":
        return WorldModelState(
            h=self.h.to(device),
            z=self.z.to(device),
            prior_stats=None if self.prior_stats is None else {k: v.to(device) for k, v in self.prior_stats.items()},
            post_stats=None if self.post_stats is None else {k: v.to(device) for k, v in self.post_stats.items()},
        )

    def clone(self) -> "WorldModelState":
        return WorldModelState(
            h=self.h.clone(),
            z=self.z.clone(),
            prior_stats=None if self.prior_stats is None else {k: v.clone() for k, v in self.prior_stats.items()},
            post_stats=None if self.post_stats is None else {k: v.clone() for k, v in self.post_stats.items()},
        )

    def detach(self) -> "WorldModelState":
        return WorldModelState(
            h=self.h.detach(),
            z=self.z.detach(),
            prior_stats=None if self.prior_stats is None else {k: v.detach() for k, v in self.prior_stats.items()},
            post_stats=None if self.post_stats is None else {k: v.detach() for k, v in self.post_stats.items()},
        )


class WorldModel(nn.Module):
    """
    Dreamer‑V3 world model:

      • ObsEncoder: symlog‑MLP encoder
      • RSSMCore: deterministic transition h_{t+1} = f(h_t, a_t)
      • Prior / Posterior: factored discrete latents over K×C
      • ObsDecoder: reconstructs observations from (h, z)
      • RewardHead: distributional symlog reward
      • ContinueHead: distributional continuation (episode continues vs terminates)
      • KL: structured KL_dyn / KL_rep with per‑factor free‑nats
    """

    def __init__(
        self,
        *,
        obs_space: gym.Space,
        latent: LatentConfig,
        net: NetworkConfig,
        free_nats: float = 3.0,
        kl_cfg: Optional[KLConfig] = None,
        aux_objectives=None,
        device: Optional[torch.device] = None,
        deterministic_latent_for_tests: bool = False,
        probe=None,
    ):
        super().__init__()

        self.device = device or torch.device("cpu")
        self.latent = latent
        self.net_cfg = net
        self.free_nats = free_nats
        self.probe = probe

        self.kl_cfg = kl_cfg or KLConfig(
            max_kl=100.0,
            min_kl=-1e-6,
            require_nonzero=True,
        )

        self.obs_space = obs_space
        self.flat_obs_dim = get_flat_obs_dim(obs_space)
        self.embed_size = net.hidden_size

        self.encoder = build_obs_encoder(obs_space, embed_dim=self.embed_size).to(self.device)
        self.rssm: RSSMCore = RSSMCore(latent=latent, net=net).to(self.device)

        self.prior: Prior = Prior(
            latent=latent, net=net, deterministic_latent_for_tests=deterministic_latent_for_tests
        ).to(self.device)

        self.posterior: Posterior = Posterior(
            latent=latent, net=net, deterministic_latent_for_tests=deterministic_latent_for_tests
        ).to(self.device)

        self.decoder: ObsDecoder = ObsDecoder(latent=latent, net=net, output_dim=self.flat_obs_dim).to(self.device)

        self.reward_heads = MultiRewardHead(
            latent=latent,
            net=net,
            num_aux=0,
        ).to(self.device)

        self.continue_head: ContinueHead = ContinueHead(latent=latent, net=net).to(self.device)

        # -------------------------------------------------------------
        # Auxiliary heads (novelty, reachability, affordance, skill, resource)
        # -------------------------------------------------------------
        all_aux_heads = make_aux_heads(
            deter_size=latent.deter_size,
            z_dim=latent.z_dim,
            action_dim=net.action_dim or 0,
            num_skills=net.action_dim or 0,
            hidden=net.hidden_size,
        )

        # Filter by config
        self.aux_heads = nn.ModuleDict({cfg.name: all_aux_heads[cfg.name] for cfg in (aux_objectives or [])}).to(
            self.device
        )

        self.aux_objectives = aux_objectives or []

        # Backward‑compatibility alias for invariants + actor/critic tests
        self.reward_head = self.reward_heads.main

    def init_state(self, batch_size: int) -> WorldModelState:
        device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.latent.deter_size, device=device)
        z0 = torch.zeros(batch_size, self.latent.num_classes, self.latent.stoch_size, device=device)

        return WorldModelState(h=h0, z=z0)

    def observe_step(
        self,
        prev_state: Any,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        is_first: Optional[torch.Tensor] = None,
        is_last: Optional[torch.Tensor] = None,
        is_terminal: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        One environment step:

          • encode obs_t
          • posterior q(z_t | h_{t-1}, embed_t)
          • prior    p(z_t | h_{t-1})
          • deterministic transition h_t = f(h_{t-1}, a_{t-1})
          • decode obs_t, reward_t, continue_t
          • compute structured KL with free‑nats
        """
        prev_state = self._ensure_state(prev_state)
        embed = self.encoder(obs)

        post_stats = self.posterior(prev_state.h, embed)
        prior_stats = self.prior(prev_state.h)

        z = post_stats["z"]
        h = self.rssm(prev_state.h, action)

        post_stats = {**post_stats, "h": h}
        prior_stats = {**prior_stats, "h": prev_state.h}

        post = WorldModelState(h=h, z=z, prior_stats=prior_stats, post_stats=post_stats)
        prior = WorldModelState(h=prev_state.h, z=prior_stats["z"], prior_stats=prior_stats, post_stats=None)

        recon = self.decoder(h, z)
        reward_main_logits, reward_aux_logits = self.reward_heads(h, z)
        cont_logits = self.continue_head(h, z)

        aux_logits = {name: head(h, z) for name, head in self.aux_heads.items()}

        kl_dict = structured_kl(
            q_probs=post_stats["probs"],
            p_probs=prior_stats["probs"],
            free_nats=self.free_nats,
            kl_cfg=self.kl_cfg,
        )

        if self.probe:
            self.probe.wm_observe(embed, post_stats, prior_stats)
            self.probe.wm_kl(kl_dict)

        for key in ["kl_dyn", "kl_rep", "kl_total"]:
            if not torch.isfinite(kl_dict[key]).all():
                raise ValueError(f"KL divergence {key} is not finite: {kl_dict[key]}")
            post_stats[key] = kl_dict[key]

        # NOTE:
        #   We return `reward_logits` for backward‑compatibility with the Dreamer‑V3 API and test suite. The training
        #   pipeline (imagination, actor‑critic update, value learning) all expect a single key named `reward_logits`
        #   that contains the *main* reward‑head logits.
        #
        #   Even though this model now supports multiple reward heads (`reward_main_logits`, `reward_aux_logits`), the
        #   legacy key must remain so downstream code continues to work without modification.
        #
        #   In short: `reward_logits` is an alias for the main reward head.
        return {
            "post": post,
            "prior": prior,
            "post_stats": post_stats,
            "prior_stats": prior_stats,
            "recon": recon,
            "reward_logits": reward_main_logits,
            "reward_main_logits": reward_main_logits,
            "reward_aux_logits": reward_aux_logits,
            "cont_logits": cont_logits,
            "aux_logits": aux_logits,
            "kl": post_stats["kl_total"],
        }

    def imagine_step(
        self,
        prev: Any,
        actor: nn.Module,
        deterministic_imagination: bool = False,
    ) -> WorldModelState:
        """
        Imagination step in latent space:

          • actor(h, z) → logits over actions
          • sample or argmax action
          • one‑hot encode action
          • RSSMCore transition h_{t+1}
          • prior p(z_{t+1} | h_{t+1})
        """
        prev_state = self._ensure_state(prev)

        logits = actor(prev_state.h, prev_state.z)
        a = act_in_imagination(logits, deterministic_imagination=deterministic_imagination)

        assert self.net_cfg.action_dim is not None, "action_dim must be specified in net config for imagine_step"
        action = F.one_hot(a, num_classes=self.net_cfg.action_dim).float()

        h = self.rssm(prev_state.h, action)
        prior = self.prior(h)

        """
        # Non-determinism in imagination shows up after the RSSM transition:
        - actor logits may diverge
        - sampled actions may diverge
        - RSSM transition may diverge
        - prior distribution may diverge
        """
        if self.probe:
            self.probe.wm_imagine(logits, action, h)

        if deterministic_imagination:
            idx = prior["probs"].argmax(dim=-1)
            z = F.one_hot(idx, num_classes=self.latent.stoch_size).float()
        else:
            z = prior["z"]

        return WorldModelState(h=h, z=z, prior_stats=prior, post_stats=None)

    def _ensure_state(self, s: Any) -> WorldModelState:
        if isinstance(s, WorldModelState):
            return s
        if isinstance(s, dict) and "state" in s:
            return s["state"]
        raise TypeError("State must be WorldModelState or dict with 'state'")

    def imagine_trajectory_for_training(self, actor, critic, start_state, horizon, deterministic_imagination=False):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_training

        return imagine_trajectory_for_training(self, actor, critic, start_state, horizon, deterministic_imagination)

    def imagine_trajectory_for_testing(self, actor, start_state, horizon):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_testing

        return imagine_trajectory_for_testing(self, actor, start_state, horizon)
