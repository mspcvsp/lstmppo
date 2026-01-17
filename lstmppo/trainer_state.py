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
from .types import Config, PolicyUpdateInfo, EpisodeStats
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
    ep_len_history: list
    ep_return_history: list
    kl_history: list
    entropy_history: list
    ev_history: list

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
        
        self.ep_len_history = []
        self.ep_return_history = []
        self.kl_history = []
        self.entropy_history = []
        self.ev_history = []
        self.policy_drift_history = []
        self.value_drift_history = []
        self.prev_h_norm = None
        self.prev_c_norm = None
        self.h_norm_history = []
        self.c_norm_history = []
        self.h_drift_history = []
        self.c_drift_history = []

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
        self.stats["policy_drift"] += upd.policy_drift.detach()
        self.stats["value_drift"] += upd.value_drift.detach()
        
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
        self.stats["h_norm"] += upd.h_norm.detach()
        self.stats["c_norm"] += upd.c_norm.detach()
        self.stats["h_drift"] += upd.h_drift.detach()
        self.stats["c_drift"] += upd.c_drift.detach()

        self.kl_history.append(upd.approx_kl.item())
        self.entropy_history.append(upd.entropy.item())
        self.ev_history.append(self.stats.get("explained_var", 0.0))

        self.policy_drift_history.append(upd.policy_drift.item())
        self.value_drift_history.append(upd.value_drift.item())
        self.h_norm_history.append(upd.h_norm.item())
        self.c_norm_history.append(upd.c_norm.item())
        self.h_drift_history.append(upd.h_drift.item())
        self.c_drift_history.append(upd.c_drift.item())

        if (len(self.policy_drift_history) >
            self.cfg.trainer.max_sparkline_history):
            
            self.policy_drift_history.pop(0)

        if (len(self.value_drift_history) >
            self.cfg.trainer.max_sparkline_history):
            
            self.value_drift_history.pop(0)

        if len(self.h_norm_history) > self.cfg.trainer.max_sparkline_history:
            self.h_norm_history.pop(0)

        if len(self.c_norm_history) > self.cfg.trainer.max_sparkline_history:
            self.c_norm_history.pop(0)

        if len(self.h_drift_history) > self.cfg.trainer.max_sparkline_history:
            self.h_drift_history.pop(0)

        if len(self.c_drift_history) > self.cfg.trainer.max_sparkline_history:
            self.c_drift_history.pop(0)

        if len(self.kl_history) > self.cfg.trainer.max_sparkline_history:
            self.kl_history.pop(0)
        
        if len(self.entropy_history) > self.cfg.trainer.max_sparkline_history:
            self.entropy_history.pop(0)
        
        if len(self.ev_history) > self.cfg.trainer.max_sparkline_history:
            self.ev_history.pop(0)

        self.stats["steps"] += 1

    def update_episode_stats(self,
                             ep_stats: EpisodeStats):

        self.stats["episodes"] = ep_stats.episodes
        self.stats["alive_envs"] = ep_stats.alive_envs
        
        self.stats["max_ep_len"] = ep_stats.max_ep_len
        self.stats["avg_ep_len"] = ep_stats.avg_ep_len

        self.stats["max_ep_returns"] = ep_stats.max_ep_returns
        self.stats["avg_ep_returns"] = ep_stats.avg_ep_returns

        # ---- Sparkline history tracking ----
        self.ep_len_history.append(ep_stats.avg_ep_len)
        self.ep_return_history.append(ep_stats.avg_ep_returns)

        if len(self.ep_len_history) > self.cfg.trainer.max_sparkline_history:
            self.ep_len_history.pop(0)

        if len(self.ep_return_history) > self.cfg.trainer.max_sparkline_history:
            self.ep_return_history.pop(0)

        # ---- Exponential Moving Average ----
        ema_alpha = self.cfg.trainer.avg_ep_stat_ema_alpha

        self.stats["avg_ep_len_ema"] = (
            ema_alpha * self.stats.get("avg_ep_len_ema",
                                       ep_stats.avg_ep_len) +
            (1.0 - ema_alpha) * ep_stats.avg_ep_len
        )

        self.stats["avg_ep_returns_ema"] = (
            ema_alpha * self.stats.get("avg_ep_returns_ema",
                                       ep_stats.avg_ep_returns) +
            (1.0 - ema_alpha) * ep_stats.avg_ep_returns
        )

    def compute_average_stats(self):

        if self.stats["steps"] > 0:

            norm_factor = 1.0 / self.stats["steps"]

            for key in [key for key in self.stats.keys()
                        if key not in ["steps",
                                       "episodes",
                                       "alive_envs",
                                       "max_ep_len",
                                       "avg_ep_len",
                                       "avg_ep_len_ema",
                                       "max_ep_returns",
                                       "avg_ep_returns",
                                       "avg_ep_returns_ema"]]:

                self.stats[key] *= norm_factor

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
        kl = float(self.stats.get("approx_kl", 0.0))
        target = float(self.target_kl)

        # Only activate after warmup
        if self.update_idx < 10:
            return

        # If KL is too high, reduce LR and clip range
        if kl > 3.0 * target:
            self.lr *= 0.9
            self.clip_range *= 0.9
            self.stats["kl_watchdog_triggered"] = 1

        # If KL is too low, increase clip range slightly
        elif kl < 0.3 * target:
            self.clip_range *= 1.05
            self.stats["kl_watchdog_triggered"] = 1
        else:
            self.stats["kl_watchdog_triggered"] = 0

        # Clamp clip range to safe bounds
        self.clip_range = float(
            torch.clamp(torch.tensor(self.clip_range),
                        0.05,
                        0.3)
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
    
    def init_stats(self):

        self.stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
            "policy_drift": 0.0,
            "value_drift": 0.0,
            "h_norm": 0.0,
            "c_norm": 0.0,
            "h_drift": 0.0,
            "c_drift": 0.0,
            "explained_var": 0.0,
            "steps": 0,
        }

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
            self.stats["entropy_adjusted"] = 0
            self.stats["entropy_up"] = 0
            self.stats["entropy_down"] = 0

            # Adaptive adjustment based on KL
            kl = float(self.stats.get("approx_kl", 0.0))
            target = float(self.target_kl)

            if self.update_idx > 10:  # warmup
                
                if kl < 0.5 * target:

                    self.entropy_coef *= 1.02
                    self.stats["entropy_adjusted"] = 1
                    self.stats["entropy_up"] = 1

                elif kl > 2.0 * target:
    
                    self.entropy_coef *= 0.98
                    self.stats["entropy_adjusted"] = 1
                    self.stats["entropy_down"] = 1

            # Clamp entropy coefficient
            self.entropy_coef = float(
                torch.clamp(torch.tensor(self.entropy_coef),
                            1e-4,
                            1.0)
            )

            # Log the delta (optional but very useful)
            self.stats["entropy_delta"] =\
                float(self.entropy_coef - old_entropy)
            
            self.stats["entropy_scheduled"] = float(scheduled)
        else:
            self.entropy_coef = self.cfg.sched.start_entropy_coef

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

    @property
    def warmup_complete(self):
        # Use the same warmup_updates as your LR scheduler
        return self.update_idx >= self._lr_sch.warmup_updates
        
    def render_dashboard(self):
        s = self.stats

        # PPO metrics panel
        ppo_text = Text()
        
        ppo_text.append(f" Return:     {s['avg_ep_returns']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" Length:     {s['avg_ep_len']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" Entropy:    {s['entropy']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" KL:         {s['approx_kl']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" ClipFrac:   {s['clip_frac']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" GradNorm:   {s['grad_norm']:.1f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" ExplainedV: {s['explained_var']:.3e}\n",
                        style="bold yellow")
        
        ppo_text.append(f" LR:         {self.lr:.2e}\n",
                        style="bold yellow")
        
        ppo_text.append(f" EntCoef:    {self.entropy_coef:.2e}\n",
                        style="bold yellow")
        
        ppo_text.append(f" ClipRange:  {self.clip_range:.2e}\n",
                        style="bold yellow")
                
        ppo_text.append(f" PolDrift:  {s['policy_drift']:.3e}\n",
                        style="bold yellow")

        ppo_text.append(f" ValDrift:  {s['value_drift']:.3e}\n",
                        style="bold yellow")

        ppo_text.append(f" h-norm:     {s['h_norm']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" c-norm:     {s['c_norm']:.3f}\n",
                        style="bold yellow")
        
        ppo_text.append(f" h-drift:    {s['h_drift']:.3e}\n",
                        style="bold yellow")
        
        ppo_text.append(f" c-drift:    {s['c_drift']:.3e}\n",
                        style="bold yellow")

        ppo_text.append("\n Return Trend: ",
                        style="bold cyan")
        
        ppo_text.append(sparkline(self.ep_return_history),
                        style="cyan")

        ppo_text.append("\n KL Trend:    ", style="bold cyan")
        ppo_text.append(sparkline(self.kl_history,
                                  width=20),
                                  style="cyan")

        ppo_text.append("\n Entropy Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.entropy_history,
                                  width=20),
                                  style="magenta")

        ppo_text.append("\n EV Trend:    ", style="bold cyan")
        ppo_text.append(sparkline(self.ev_history,
                                  width=20),
                                  style="green")

        ppo_text.append("\n PolDrift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.policy_drift_history),
                        style="cyan")

        ppo_text.append("\n ValDrift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.value_drift_history),
                        style="green")

        ppo_text.append("\n h-norm Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.h_norm_history),
                        style="cyan")

        ppo_text.append("\n c-norm Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.c_norm_history),
                        style="cyan")

        ppo_text.append("\n h-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.h_drift_history),
                        style="green")

        ppo_text.append("\n c-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.c_drift_history),
                        style="green")

        ppo_panel = Panel(ppo_text,
                          title="PPO Metrics",
                          border_style="bright_blue")

        # Episode stats panel
        ep_text = Text()

        ep_text.append(f" Episodes:   {s['episodes']}\n",
                       style="bold green")

        ep_text.append(f" AliveEnv:   {s['alive_envs']}\n",
                       style="bold green")
        
        ep_text.append(f" MaxEpLen:   {s['max_ep_len']:.1f}\n",
                       style="bold green")
        
        ep_text.append(f" AvgEpLen:   {s['avg_ep_len']:.1f}\n",
                       style="bold green")
        
        ep_text.append(f" EMA Len:    {s['avg_ep_len_ema']:.1f}\n",
                       style="bold green")
        
        ep_text.append(f" MaxReturn:  {s['max_ep_returns']:.2f}\n",
                       style="bold green")
        
        ep_text.append(f" AvgReturn:  {s['avg_ep_returns']:.2f}\n",
                       style="bold green")
        
        ep_text.append(f" EMA Return: {s['avg_ep_returns_ema']:.2f}\n",
                       style="bold green")
        
        ep_text.append("\n Length Trend: ",
                       style="bold cyan")
        
        ep_text.append(sparkline(self.ep_len_history),
                       style="cyan")

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


def sparkline(data, width=20):
    if len(data) == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(data), max(data)
    if mx - mn < 1e-8:
        return blocks[0] * min(len(data), width)
    scaled = [(x - mn) / (mx - mn) for x in data[-width:]]
    idx = [int(s * (len(blocks) - 1)) for s in scaled]
    return "".join(blocks[i] for i in idx)