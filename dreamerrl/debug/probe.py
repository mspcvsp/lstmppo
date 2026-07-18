# dreamerrl/debug/probe.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch


class DeterminismProbe:
    """
    Unified determinism probe module.
    Logs summary statistics only (mean/std) with subsystem tags.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, every_n: int = 1):
        self.log: Optional[logging.Logger] = logger
        self.every_n = every_n
        self.step = 0

    def _should_log(self) -> bool:
        self.step += 1
        return self.log is not None and (self.step % self.every_n == 0)

    # ---------------------------------------------------------
    # Environment probes
    # ---------------------------------------------------------
    def env_step(self, obs, reward, terminated, truncated, is_last):
        if not self._should_log():
            return

        assert self.log is not None
        log: logging.Logger = self.log  # type narrowing
        log.debug(
            f"[ENV] obs_mean={obs.mean().item():.6f} "
            f"rew_mean={reward.mean().item():.6f} "
            f"term={terminated.tolist()} trunc={truncated.tolist()} last={is_last.tolist()}"
        )

    def env_reset(self, obs):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(f"[ENV_RESET] obs_mean={obs.mean().item():.6f}")

    # ---------------------------------------------------------
    # Replay buffer probes
    # ---------------------------------------------------------
    def replay_sample(self, seed: int, ep_idx: int, ep_len: int, start: int):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(f"[REPLAY] seed={seed} ep_idx={ep_idx} ep_len={ep_len} start={start}")

    def replay_finalize(self, idx: int, length: int, total_size: int):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(f"[EP_FINALIZE] idx={idx} len={length} total_size={total_size}")

    # ---------------------------------------------------------
    # World model probes
    # ---------------------------------------------------------
    def wm_observe(self, embed: torch.Tensor, post_stats: Dict[str, Any], prior_stats: Dict[str, Any]):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(
            f"[WM_OBS] embed_mean={embed.mean().item():.6f} "
            f"post_h_mean={post_stats['h'].mean().item():.6f} "
            f"prior_h_mean={prior_stats['h'].mean().item():.6f}"
        )

    def wm_kl(self, kl_dict: Dict[str, torch.Tensor]):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(
            f"[WM_KL] dyn={kl_dict['kl_dyn'].mean().item():.6f} "
            f"rep={kl_dict['kl_rep'].mean().item():.6f} "
            f"total={kl_dict['kl_total'].mean().item():.6f}"
        )

    def wm_imagine(self, logits: torch.Tensor, action: torch.Tensor, h: torch.Tensor):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(
            f"[WM_IMAGINE] logits_mean={logits.mean().item():.6f} "
            f"action_mean={action.mean().item():.6f} "
            f"h_mean={h.mean().item():.6f}"
        )

    # ---------------------------------------------------------
    # Actor / Critic probes
    # ---------------------------------------------------------
    def actor(self, logits: torch.Tensor):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(f"[ACTOR] logits_mean={logits.mean().item():.6f} logits_std={logits.std().item():.6f}")

    def critic(self, values: torch.Tensor):
        if not self._should_log():
            return
        assert self.log is not None
        log: logging.Logger = self.log
        log.debug(f"[CRITIC] value_mean={values.mean().item():.6f} value_std={values.std().item():.6f}")
