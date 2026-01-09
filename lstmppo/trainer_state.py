from dataclasses import dataclass
import io
import json
import torch
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from .types import Config, PolicyUpdateInfo
from .learning_sch import EntropySchdeduler, LearningRateScheduler


@dataclass
class TrainerState:
    update_idx: int
    lr: float
    entropy_coef: float
    clip_range: float
    target_kl: float
    early_stopping_kl: float
    stats: dict
    writer: SummaryWriter
    jsonl_file: str
    jsonl_fp: io.TextIOWrapper
    validation_mode: bool


    def __init__(self,
                 cfg: Config,
                 validation_mode: bool = False):

        self.cfg = cfg
        self.validation_mode = validation_mode

        if self.validation_mode:

            self.cfg.env.num_envs = 1
            self.cfg.trainer.mini_batch_envs = 1
            
            self.cfg.trainer.tbptt_chunk_len =\
                self.cfg.trainer.rollout_steps  # full sequence
            
            self.cfg.sched.anneal_entropy_flag = False
            self.cfg.sched.start_entropy_coef = 0.0
            self.cfg.sched.end_entropy_coef = 0.0
            self.cfg.ppo.initial_clip_range = 0.0
            self.cfg.ppo.target_kl = 1e9  # disable early stopping
            self.cfg.trainer.debug_mode = True  # disable DropConnect

        self.clip_range = cfg.ppo.initial_clip_range
 
        self.early_stopping_kl =\
            self.cfg.ppo.target_kl * cfg.ppo.early_stopping_kl_factor

        self._entropy_sch = EntropySchdeduler(cfg)
        self._lr_sch = LearningRateScheduler(cfg)

        tb_logdir = Path(*[cfg.log.tb_logdir,
                           cfg.log.run_name])

        self.jsonl_fp = None

        jsonl_path = Path(cfg.log.jsonl_path)

        if jsonl_path.exists() is False:

            jsonl_path.mkdir(parents=True,
                             exist_ok=True)

        self.jsonl_file =\
            jsonl_path.joinpath(cfg.log.run_name + ".json") 

        self.writer = SummaryWriter(log_dir=tb_logdir)

    def reset(self,
              total_updates: int):

        self.update_idx = 0
        self._entropy_sch.reset(total_updates)
        self._lr_sch.reset(total_updates)

    def update_stats(self,
                     upd: PolicyUpdateInfo):

        self.stats["policy_loss"] += upd.policy_loss.detach()
        self.stats["value_loss"] += upd.value_loss.detach()
        self.stats["entropy"] += upd.entropy.detach()
        self.stats["approx_kl"] += upd.approx_kl.detach()
        self.stats["clip_frac"] += upd.clip_frac.detach()
        self.stats["grad_norm"] += upd.grad_norm
        self.stats["steps"] += 1

    def compute_average_stats(self):

        if self.stats["steps"] > 0:

            norm_factor = 1.0 / self.stats["steps"]

            for key in [key for key in self.stats.keys()
                        if key not in ["steps"]]:

                self.stats[key] *= norm_factor

    @property
    def global_step(self) -> int:
        return (
            self.update_idx *
            self.cfg.trainer.rollout_steps *
            self.cfg.env.num_envs
        )

    def log_metrics(self):

        for key, value in self.stats.items():

            self.writer.add_scalar(key,
                                   value,
                                   self.global_step)

        record = {k: to_float(v) for k, v in self.stats.items()}

        record["update"] = self.update_idx
        record["lr"] = float(self.lr)
        record["entropy_coef"] = float(self.entropy_coef)
        record["clip_range"] = float(self.clip_range)

        self.jsonl_fp.write(json.dumps(record) + "\n")
        self.jsonl_fp.flush()

        print(
            f"upd {self.update_idx:04d} | "
            f"pol {self.stats['policy_loss']:.3f} | "
            f"val {self.stats['value_loss']:.3f} | "
            f"ent {self.stats['entropy']:.3f} | "
            f"kl {self.stats['approx_kl']:.4f} | "
            f"clip {self.stats['clip_frac']:.3f} | "
            f"ev {self.stats['explained_var']:.3f} | "
            f"grad {self.stats['grad_norm']:.2f} | "
            f"clip_range {self.clip_range:.3f}"
        )
    
    def init_stats(self):

        self.stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
            "explained_var": 0.0,
            "steps": 0,
        }

    def apply_schedules(self,
                        optimizer: torch.optim.Adam):

        self.entropy_coef = self._entropy_sch(self.update_idx)
        self.lr = self._lr_sch(self.update_idx)

        self.writer.add_scalar("entropy_coef",
                                self.entropy_coef,
                                self.update_idx)

        self.writer.add_scalar("learning_rate",
                               self.lr,
                               self.update_idx)

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr

    def adapt_clip_range(self):

        if self.cfg.trainer.debug_mode is False:

            avg_kl = self.stats["approx_kl"]

            if avg_kl > 2.0 * self.cfg.ppo.target_kl:
                self.clip_range *= 0.9
            elif avg_kl < 0.5 * self.cfg.ppo.target_kl:
                self.clip_range *= 1.05

            self.clip_range = float(torch.clamp(
                torch.tensor(self.clip_range),
                0.05,
                0.3
            ))

    def should_stop_early(self):

        stop_early = (
            self.stats["approx_kl"] > self.early_stopping_kl
            and self.cfg.trainer.debug_mode is False
        )

        return stop_early

    def should_save_checkpoint(self):

        return (
            self.update_idx % 
            self.cfg.trainer.updates_per_checkpoint == 0
        )
    

def to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)