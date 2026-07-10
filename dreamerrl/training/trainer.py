from __future__ import annotations

import math
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.pyplot import step
from torch.utils.tensorboard import SummaryWriter

import wandb
from dreamerrl.env.popgym.popgym_parallel_env import PopGymParallelEnv
from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.training.core import actor_critic_update, world_model_training_step
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import DreamerConfig, LatentConfig, LRScheduleConfig, NetworkConfig, WorldModelMetrics

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Always-defined module-level logger alias
log = logger


class CosineWarmupScheduler:
    """Single shared LR schedule for world, actor, critic (Dreamer‑V3 requirement)."""

    def __init__(self, cfg: LRScheduleConfig):
        self.cfg = cfg

    def __call__(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.base_lr * (step / self.cfg.warmup_steps)

        progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.total_steps - self.cfg.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        min_lr = self.cfg.base_lr * self.cfg.lr_floor
        return min_lr + (self.cfg.base_lr - min_lr) * cosine


class DreamerTrainer:
    """
    Dreamer‑V3 trainer.

    Wiring summary:
      • WorldModel: factored discrete latent RSSM + symlog reward/continue heads
      • Actor: policy over discrete actions from (h, z)
      • Critic: distributional value head over symlog bins
      • ReplayBuffer: stores raw env transitions, samples (B, L, ·) sequences
    """

    def __init__(self, cfg: DreamerConfig):
        self.cfg = cfg
        self.sample_step = 0

        if self.cfg.train.enable_repro_log:
            # Remove default stderr handler
            log.remove()

            # Bind seeds for reproducibility
            bound = log.bind(
                train_seed=self.cfg.train.seed,
                env_seed=self.cfg.env.seed,
            )

            # Add deterministic file sink
            bound.add(
                os.path.join(LOG_DIR, f"repro_seed_{self.cfg.train.seed}.log"),
                format="TRAIN={train_seed} ENV={env_seed} | {message}",
                level="DEBUG",
                mode="w",
                enqueue=False,
            )

            # Save bound logger on trainer instance
            self.log = bound
        else:
            self.log = log

        logdir = os.path.join(cfg.log.tb_logdir, cfg.log.run_name)
        self.tb = SummaryWriter(log_dir=logdir)

        # -----------------------------------------------------
        # Seeding (must be first for reproducibility)
        # -----------------------------------------------------
        set_global_seeds(cfg.train.seed)

        # Ensure bit-for-bit reproducibility across PyTorch versions and hardware.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if self.cfg.train.collect_steps < self.cfg.env.max_episode_steps:
            print(
                f"[Warning] collect_steps ({self.cfg.train.collect_steps}) is smaller than "
                f"max_episode_steps ({self.cfg.env.max_episode_steps}). "
                "Episodes will not finish inside a single rollout. "
                "Replay buffer will remain empty unless collect_env_steps() "
                "is called multiple times before sampling."
            )

        self.device = torch.device("cuda" if cfg.train.cuda and torch.cuda.is_available() else "cpu")

        # -----------------------------------------------------
        # Environment
        # -----------------------------------------------------
        if cfg.env.parallel:
            self.env = PopGymParallelEnv(cfg.env, device=self.device)
        else:
            self.env = PopGymVecEnv(cfg.env, cfg.train.enable_repro_log, device=self.device)

        obs_space = self.env.venv.single_observation_space
        self.action_dim = self.env.action_dim

        # -----------------------------------------------------
        # Latent + Network configs
        # -----------------------------------------------------
        latent = LatentConfig(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            num_classes=cfg.world.num_classes,
        )

        net_world = NetworkConfig(
            hidden_size=cfg.world.hidden_size,
            action_dim=self.action_dim,
            value_bins=cfg.world.value_bins,
        )

        # -----------------------------------------------------
        # World Model
        # -----------------------------------------------------
        self.world = WorldModel(
            obs_space=obs_space,
            latent=latent,
            net=net_world,
            free_nats=cfg.world.free_nats,
            num_aux_reward_heads=cfg.world.num_aux_reward_heads,
            device=self.device,
        )
        self.world_state = self.world.init_state(self.env.batch_size)
        self.world_state = self.world_state.to(self.device)

        # -----------------------------------------------------
        # Actor + Critic
        # -----------------------------------------------------
        net_actor = NetworkConfig(
            hidden_size=cfg.ac.actor_hidden,
            action_dim=self.action_dim,
        )

        net_critic = NetworkConfig(
            hidden_size=cfg.ac.critic_hidden,
            value_bins=cfg.world.value_bins,
        )

        self.actor = Actor(latent=latent, net=net_actor).to(self.device)
        self.critic = ValueHead(latent=latent, net=net_critic).to(self.device)

        # -----------------------------------------------------
        # Replay Buffer
        # -----------------------------------------------------
        flat_obs_dim = self.world.flat_obs_dim
        self.replay = ReplayBuffer(
            cfg=cfg.train,
            obs_dim=flat_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        # -----------------------------------------------------
        # Optimizers
        # -----------------------------------------------------
        self.model_opt = torch.optim.Adam(self.world.parameters(), lr=cfg.train.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.train.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.critic_lr)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        if cfg.log.enable_wandb:
            wandb.init(project="dreamer_v3", config=cfg.__dict__)

        self.env_state: Dict[str, Any] = self.env.reset()
        self.total_env_steps: int = 0

        # NOTE:
        # Dreamer requires seq_len <= collect_steps <= max_episode_steps.
        # This clamp MUST run at the end of __init__, after all config overrides
        # (including test overrides) have been applied. Running it earlier allows
        # stale default values to leak into the trainer and break invariants.
        self.cfg.train.seq_len = min(
            self.cfg.train.seq_len,
            self.cfg.train.collect_steps,
            self.cfg.env.max_episode_steps,
        )

    @property
    def global_step(self) -> int:
        return self.total_env_steps

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    def train(self, total_updates: int) -> None:
        lr_cfg = LRScheduleConfig(
            base_lr=self.cfg.train.model_lr,
            warmup_steps=self.cfg.train.warmup_steps,
            total_steps=total_updates,
            lr_floor=0.1,
        )
        lr_schedule = CosineWarmupScheduler(lr_cfg)

        self.recent_returns = []

        for update_idx in range(total_updates):
            t0 = time.time()

            lr = lr_schedule(update_idx)
            for pg in self.model_opt.param_groups:
                pg["lr"] = lr
            for pg in self.actor_opt.param_groups:
                pg["lr"] = lr
            for pg in self.critic_opt.param_groups:
                pg["lr"] = lr

            self.collect_env_steps()

            batch = self.replay.sample(
                batch_size=self.cfg.train.batch_size,
                seed=self.cfg.train.seed + self.sample_step,
            )
            self.sample_step += 1

            wm_metrics = self.update_world_model(batch, update_idx)
            actor_loss, critic_loss = self.update_actor_critic(batch, update_idx)

            ep_return = 0.0
            if self.env_state["is_last"].any():
                ep_return = self.env_state["reward"].sum().item()
                self.tb.add_scalar("env/ep_return", ep_return, step)

                self.recent_returns.append(ep_return)
                self.recent_returns = self.recent_returns[-50:]
                self.tb.add_scalar("env/avg_return_50", np.mean(self.recent_returns), step)

            if self.cfg.log.enable_wandb:
                wandb.log(
                    {
                        "loss/model": wm_metrics.total_loss.item(),
                        "loss/actor": actor_loss,
                        "loss/critic": critic_loss,
                        "time/update": time.time() - t0,
                    },
                    step=update_idx,
                )

    # -------------------------------------------------------------
    # Collect steps from environment
    # -------------------------------------------------------------
    def collect_env_steps(self) -> None:
        """
        Collect multiple environment steps per call.
        Dreamer-V3 requires enough steps to finish episodes so the replay buffer
        can finalize and store them.
        """
        for _ in range(self.cfg.train.collect_steps):
            # 1. Choose discrete action
            if self.cfg.train.deterministic_env:
                # Fully deterministic for reproducibility tests
                actions_discrete = self.actor(self.world_state.h, self.world_state.z).argmax(dim=-1)
            else:
                # Normal Dreamer training behavior
                if self.global_step < self.cfg.train.random_exploration_steps:
                    actions_discrete = torch.randint(
                        low=0,
                        high=self.action_dim,
                        size=(self.env.batch_size,),
                        device=self.device,
                    )
                else:
                    actions_discrete, _ = self.actor.act(self.world_state)

            if self.cfg.train.enable_repro_log:
                self.log.debug(f"ACTION {self.global_step}: {actions_discrete.tolist()}")

            # 2. Step environment
            env_out = self.env.step(actions_discrete)

            if self.cfg.train.enable_repro_log:
                self.log.debug(f"REWARD {self.global_step}: {env_out['reward'].tolist()}")

            self._check_consistency(env_out)

            # Move env outputs to CUDA
            for k, v in env_out.items():
                if torch.is_tensor(v):
                    env_out[k] = v.to(self.device)

            # 3. One-hot encode actions for RSSMCore
            actions_one_hot = F.one_hot(actions_discrete, num_classes=self.action_dim).float()

            # 4. Update latent state
            wm_out = self.world.observe_step(
                prev_state=self.world_state,
                obs=env_out["state"],
                action=actions_one_hot,
                reward=env_out["reward"],
                is_first=env_out["is_first"],
                is_last=env_out["is_last"],
                is_terminal=env_out["is_terminal"],
            )

            self.world_state = wm_out["post"].to(self.device)

            # 5. Store raw transition in replay buffer
            self.replay.add(
                obs=env_out["state"],
                action=actions_discrete,
                reward=env_out["reward"],
                done=env_out["is_last"].float().to(self.device),
            )

            # 6. Update counters
            self.env_state = env_out
            self.total_env_steps += self.env.batch_size

    def _check_consistency(self, env_out: dict) -> None:
        state = env_out["state"]
        reward = env_out["reward"]
        is_last = env_out["is_last"]
        is_terminal = env_out["is_terminal"]
        prev_action = env_out.get("prev_action", None)

        # ✅ unified environment interface
        batch_size = self.env.batch_size
        obs_dim = self.env.obs_dim
        action_dim = self.env.action_dim

        if self.cfg.train.enforce_length_invariants:
            seq_len = self.cfg.train.seq_len
            max_steps = self.cfg.env.max_episode_steps
            collect_steps = self.cfg.train.collect_steps

            if seq_len > max_steps:
                raise ValueError(f"seq_len={seq_len} must be <= max_episode_steps={max_steps}")

            if seq_len > collect_steps:
                raise ValueError(f"seq_len={seq_len} must be <= collect_steps={collect_steps}")

        if state.shape != (batch_size, obs_dim):
            raise RuntimeError(f"State shape mismatch: got {state.shape}, expected ({batch_size}, {obs_dim})")

        if prev_action is not None and prev_action.shape != (batch_size, action_dim):
            raise RuntimeError(
                f"Prev_action shape mismatch: got {prev_action.shape}, expected ({batch_size}, {action_dim})"
            )

        if reward.shape != (batch_size,):
            raise RuntimeError(f"Reward shape mismatch: got {reward.shape}, expected ({batch_size},)")

        if not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite reward detected")

        # ✅ use is_first so Ruff stops complaining
        for key in ("is_first", "is_last", "is_terminal"):
            flag = env_out[key]
            if flag.shape != (batch_size,):
                raise RuntimeError(f"{key} shape mismatch: got {flag.shape}, expected ({batch_size},)")
            if flag.dtype != torch.bool:
                raise RuntimeError(f"{key} must be boolean, got {flag.dtype}")

        if is_last.any() and not is_terminal.any():
            raise RuntimeError(
                f"Environment produced early termination; Dreamer requires fixed-length episodes of {max_steps} steps"
            )

    # -------------------------------------------------------------
    # World Model Update
    # -------------------------------------------------------------
    def update_world_model(self, batch: Dict[str, torch.Tensor], update_idx: int) -> WorldModelMetrics:
        # Zero optimizer gradients
        self.model_opt.zero_grad()

        # Forward pass
        metrics = world_model_training_step(
            world_model=self.world,
            batch=batch,
            kl_scale=self.cfg.world.kl_scale,
        )

        loss = metrics.total_loss

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.train.grad_clip)
        self.model_opt.step()

        # TensorBoard logging
        step = update_idx

        self.tb.add_scalar("wm/total_loss", metrics.total_loss.item(), step)
        self.tb.add_scalar("wm/recon_loss", metrics.recon_loss.item(), step)
        self.tb.add_scalar("wm/reward_loss", metrics.reward_loss.item(), step)
        self.tb.add_scalar("wm/cont_loss", metrics.cont_loss.item(), step)
        self.tb.add_scalar("wm/kl_dyn", metrics.kl_dyn.item(), step)
        self.tb.add_scalar("wm/kl_rep", metrics.kl_rep.item(), step)

        for i, aux in enumerate(metrics.aux_losses):
            self.tb.add_scalar(f"wm/aux_loss_{i}", aux.item(), step)

        return metrics

    # -------------------------------------------------------------
    # Actor + Critic Update
    # -------------------------------------------------------------
    def update_actor_critic(self, batch: Dict[str, torch.Tensor], update_idx: int):
        actor_loss, critic_loss = actor_critic_update(
            world_model=self.world,
            actor=self.actor,
            critic=self.critic,
            batch=batch,
            imagination_horizon=self.cfg.world.imagination_horizon,
            discount=self.cfg.ac.discount,
            lam=self.cfg.ac.lambda_,
            deterministic_imagination=self.cfg.train.deterministic_imagination,
        )

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.train.grad_clip)
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.train.grad_clip)
        self.critic_opt.step()

        return float(actor_loss.item()), float(critic_loss.item())
