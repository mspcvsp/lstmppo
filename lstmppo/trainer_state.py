from dataclasses import dataclass
import io
import json
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


from torch.utils.tensorboard import SummaryWriter
from .types import Config, MetricsHistory, PolicyUpdateInfo
from .types import EpisodeStats, Metrics, MetricsHistory
from .learning_sch import EntropySchdeduler, LearningRateScheduler


@dataclass
class TrainerState:
    update_idx: int
    lr: float
    entropy_coef: float
    clip_range: float
    target_kl: float
    early_stopping_kl: float
    writer: SummaryWriter
    jsonl_file: str
    jsonl_fp: io.TextIOWrapper
    validation_mode: bool

    def __init__(self,
                 cfg: Config,
                 validation_mode: bool = False):

        self.cfg = cfg
        self.validation_mode = validation_mode
        self.metrics = Metrics()

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

        self.clip_range = self.cfg.ppo.initial_clip_range
        self.target_kl = self.cfg.ppo.target_kl
 
        self.early_stopping_kl =\
            self.cfg.ppo.target_kl * cfg.ppo.early_stopping_kl_factor

        self._entropy_sch = EntropySchdeduler(self.cfg)
        self._lr_sch = LearningRateScheduler(self.cfg)

        tb_logdir = Path(*[self.cfg.log.tb_logdir,
                           self.cfg.log.run_name])

        self.writer = SummaryWriter(log_dir=tb_logdir)

        self.jsonl_fp = None

        jsonl_path = Path(self.cfg.log.jsonl_path)
        if jsonl_path.exists() is False:

            jsonl_path.mkdir(parents=True,
                             exist_ok=True)

        self.jsonl_file =\
            jsonl_path.joinpath(self.cfg.log.run_name + ".json")

        self.prev_h_norm = None
        self.prev_c_norm = None
        self.prev_i_mean = None
        self.prev_f_mean = None
        self.prev_g_mean = None
        self.prev_o_mean = None

        self.avg_ep_len_ema: float = 0.0
        self.avg_ep_returns_ema: float = 0.0

        self.history = MetricsHistory(
            max_len=self.cfg.trainer.max_sparkline_history
        )

    def reset(self,
              total_updates: int):

        self.update_idx = 0
        self._entropy_sch.reset(total_updates)
        self._lr_sch.reset(total_updates)

    def update_stats(self,
                     upd: PolicyUpdateInfo):

        """
        LSTM hidden‑state norm drift is one of the most powerful
        diagnostics for recurrent PPO. It tells you whether your LSTM is:
        • 	exploding
        • 	collapsing
        • 	saturating
        • 	drifting across updates
        • 	or behaving consistently

        This is incredibly useful for diagnosing:
        - exploding LSTMs
        - vanishing LSTMs
        - unstable TBPTT
        - bad reward scaling
        - bad normalization
        - bad initialization
        - DropConnect issues
        """
        self.metrics.accumulate(upd)

        self.history.update(upd, self.metrics)

    def update_episode_stats(self,
                             ep_stats: EpisodeStats):

        self.metrics.update_episode_stats(ep_stats)
        
        # ---- Exponential Moving Average 
        ema_alpha = self.cfg.trainer.avg_ep_stat_ema_alpha

        self.avg_ep_len_ema = (
            ema_alpha * self.avg_ep_len_ema +
            (1.0 - ema_alpha) * ep_stats.avg_ep_len
        )

        self.avg_ep_returns_ema = (
            ema_alpha * self.avg_ep_returns_ema +
            (1.0 - ema_alpha) * ep_stats.avg_ep_returns
        )

        # ---- Sparkline history tracking ----
        self.history.push("ep_len", ep_stats.avg_ep_len)
        self.history.push("ep_return", ep_stats.avg_ep_returns)

    def compute_average_metrics(self):

        if self.metrics.steps > 0:
            self.metrics.normalize()

    @property
    def global_step(self) -> int:
        return (
            self.update_idx *
            self.cfg.trainer.rollout_steps *
            self.cfg.env.num_envs
        )

    def kl_watchdog(self):
        """
        KL is only meaningful after:
        • 	all minibatches are processed
        • 	stats are averaged
        • 	schedules are applied
        • 	clip range adaptation is applied
        """
        kl = getattr(self.metrics, "approx_kl", 0.0)
        target = float(self.target_kl)

        # Only activate after warmup
        if self.update_idx < 10:
            return

        # If KL is too high, reduce LR and clip range
        if kl > 3.0 * target:
            self.lr *= 0.9
            self.clip_range *= 0.9
            setattr(self.metrics, "kl_watchdog_triggered", 1)

        # If KL is too low, increase clip range slightly
        elif kl < 0.3 * target:
            self.clip_range *= 1.05
            setattr(self.metrics, "kl_watchdog_triggered", 1)
        else:
            setattr(self.metrics, "kl_watchdog_triggered", 0)

        # Clamp clip range to safe bounds
        self.clip_range = float(
            torch.clamp(torch.tensor(self.clip_range),
                        0.05,
                        0.3)
        )

    def log_metrics(self):

        record = self.metrics.to_dict()

        for key, value in record.items():

            self.writer.add_scalar(key,
                                   value,
                                   self.global_step)

        record["update"] = self.update_idx
        record["lr"] = float(self.lr)
        record["entropy_coef"] = float(self.entropy_coef)
        record["clip_range"] = float(self.clip_range)

        self.jsonl_fp.write(json.dumps(record) + "\n")
        self.jsonl_fp.flush()

    def init_stats(self):
        self.metrics.initialize()

    def apply_schedules(self,
                        optimizer: torch.optim.Adam):

        """
        Instead of decaying entropy on a fixed schedule, adaptive entropy 
        adjusts itself based on KL divergence:

        • 	If KL is too low → policy isn’t changing → increase entropy
        (more exploration)
        
        • 	If KL is too high → policy is changing too fast → decrease
        entropy (more caution)

        This keeps exploration self‑tuning and dramatically stabilizes POMDPs
        like Position‑Only CartPole.
        """
        if self.cfg.sched.anneal_entropy_flag:

            # Start with scheduled value
            scheduled = self._entropy_sch(self.update_idx)
            old_entropy = self.entropy_coef
            self.entropy_coef = scheduled

            # Reset logging flags
            self.metrics.entropy_adjusted = 0
            self.metrics.entropy_up = 0
            self.metrics.entropy_down = 0

            # Adaptive adjustment based on KL
            kl = getattr(self.metrics, "approx_kl", 0.0)
            target = float(self.target_kl)

            if self.update_idx > 10:  # warmup
                
                if kl < 0.5 * target:

                    self.entropy_coef *= 1.02
                    setattr(self.metrics, "entropy_adjusted", 1)
                    setattr(self.metrics, "entropy_up", 1)

                elif kl > 2.0 * target:
    
                    self.entropy_coef *= 0.98
                    setattr(self.metrics, "entropy_adjusted", 1)
                    setattr(self.metrics, "entropy_down", 1)

            # Clamp entropy coefficient
            self.entropy_coef = float(
                torch.clamp(torch.tensor(self.entropy_coef),
                            1e-4,
                            1.0)
            )

            # Log the delta (optional but very useful)
            setattr(self.metrics,
                    "entropy_delta",
                    float(self.entropy_coef - old_entropy))

            setattr(self.metrics,
                    "entropy_scheduled",
                    float(scheduled))
        else:
            self.entropy_coef = self.cfg.sched.start_entropy_coef
            
            setattr(self.metrics,
                    "entropy_scheduled",
                    self.entropy_coef)

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

            avg_kl = self.metrics["approx_kl"]

            if avg_kl > 2.0 * self.cfg.ppo.target_kl:
                self.clip_range *= 0.9
            elif avg_kl < 0.5 * self.cfg.ppo.target_kl:
                self.clip_range *= 1.05

            self.clip_range = float(torch.clamp(
                torch.tensor(self.clip_range),
                0.05,
                0.3
            ))

    @property
    def warmup_complete(self):
        # Use the same warmup_updates as your LR scheduler
        return self.update_idx >= self._lr_sch.warmup_updates
        
    def render_dashboard(self):
        
        ppo_text = self.metrics.render_ppo_metrics(self.lr,
                                                 self.entropy_coef,
                                                 self.clip_range)

        self.history.render_ppo_history(ppo_text)

        ppo_panel = Panel(ppo_text,
                          title="PPO Metrics",
                          border_style="bright_blue")

        ep_text = self.metrics.render_episode_stats(self.avg_ep_len_ema,
                                                    self.avg_ep_returns_ema)

        self.history.render_episode_history(ep_text)

        ep_panel = Panel(ep_text,
                         title="Episode Stats",
                         border_style="bright_green")

        # Layout
        layout = Layout()
        layout.split_row(
            Layout(ppo_panel, ratio=1),
            Layout(ep_panel, ratio=1),
        )

        return layout

    def should_save_checkpoint(self):

        return (
            self.update_idx % 
            self.cfg.trainer.updates_per_checkpoint == 0
        )
    

def to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)
