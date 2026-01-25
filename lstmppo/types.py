"""
• 	Config dataclasses must be pure data containers
• 	No side effects in __init__
• 	No environment creation
• 	No torch backend mutation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

import gymnasium as gym
import torch
from rich.text import Text

from .obs_encoder import get_flat_obs_dim


@dataclass
class PPOHyperparams:
    gamma: float = 0.99
    """ Discount factor"""
    gae_lambda: float = 0.95
    """ Generalized Advantage Estimate (GAE) lambda"""
    initial_clip_range: float = 0.15
    """ Initial clip range """
    update_epochs: int = 4
    """ Number of update epochs """
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping"""
    target_kl: float = 0.005
    """ Target KL divergence """
    early_stopping_kl_factor: float = 1.5
    """ Stop early if approx_kl exceeds this factor times target_kl """


@dataclass
class LSTMConfig:
    enc_hidden_size: int = 128
    """Encoder hidden size"""
    lstm_hidden_size: int = 128
    """LSTM hidden size"""
    dropconnect_p: float = 0.5
    """LSTM drop connect probability"""
    lstm_ar_coef: float = 1.0
    """LSTM activation regularization (AR)"""
    lstm_tar_coef: float = 0.5
    """LSTM temporal activation regularization (TAR)"""


@dataclass
class ScheduleConfig:
    base_lr: float = 1e-4
    """ Base learning rate """
    lr_warmup_pct: float = 10.0
    """ Learning rate warmup percentage """
    lr_final_pct: float = 10.0
    """ Final learning rate percentage of base learning rate """
    anneal_entropy_flag: bool = True
    """ Toggle entropy coefficient annealing """
    start_entropy_coef: float = 0.1
    """ Starting value of entropy coefficient for annealing """
    end_entropy_coef: float = 0.02


@dataclass
class TrainerConfig:
    cuda: bool = True
    """ Whether to use CUDA """
    torch_deterministic: bool = True
    """ Sets the value of torch.backends.cudnn.deterministic """
    rollout_steps: int = 256
    """ Horizon"""
    mini_batch_envs: int = 4
    """ Number of environments / minibatch"""
    updates_per_checkpoint: int = 10
    """ Number of updates / checkpoint """
    debug_mode: bool = False
    """ Toggles debug mode """
    seed: int = 351530767
    """seed of the experiment"""
    tbptt_chunk_len: int = 64
    """ Truncated BPTT steps """
    exp_name: str = "RLWarmup"
    """ Experiment name """
    avg_ep_stat_ema_alpha: float = 0.9
    """ Average episode statistics EMA alpha """
    max_sparkline_history: int = 100
    """ Maximum number of sparkline history points to keep """
    gate_sat_eps: float = 0.05
    """ LSTM gate saturation epsilon """
    gate_ent_eps: float = 1e-6
    """ LSTM gate entropy epsilon """


@dataclass
class LoggingConfig:
    tb_logdir: str = "./tb_logs"
    """ TensorBoard log directory """
    jsonl_path: str = "./jsonl"
    """ JSONL (one JSON object per line) log directory """
    checkpoint_dir: str = "./checkpoints"
    """ Model checkpoints directory"""
    run_name: str = ""
    """ Run Name """


@dataclass
class EnvironmentConfig:
    env_id: str = "CartPole-v1"
    """Environment identifier"""
    num_envs: int = 16
    """ Number of environments """
    obs_space: gym.Space | None = None
    """ Observation space """
    flat_obs_dim: int = 0
    """ Flattened observation dimension """
    action_dim: int = 0
    """ Action dimension """
    max_episode_steps: int = None
    """ Maximum number of steps per episode """
    max_env_history: int = 30
    """ Maximum per env history length """
    ep_len_reward_bonus: float = 0.1
    """ Episode length reward bonus """


@dataclass
class BufferConfig:
    rollout_steps: int
    num_envs: int
    mini_batch_envs: int
    gamma: float
    lam: float
    lstm_hidden_size: int


@dataclass
class Config:
    # field(default_factory=...) avoids the same instance of each sub‑config
    # from being shared across all Config() objects
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ppo: PPOHyperparams = field(default_factory=PPOHyperparams)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    sched: ScheduleConfig = field(default_factory=ScheduleConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def buffer_config(self) -> BufferConfig:
        return BufferConfig(
            rollout_steps=self.trainer.rollout_steps,
            num_envs=self.env.num_envs,
            mini_batch_envs=self.trainer.mini_batch_envs,
            gamma=self.ppo.gamma,
            lam=self.ppo.gae_lambda,
            lstm_hidden_size=self.lstm.lstm_hidden_size,
        )

    @property
    def obs_dim(self):
        return self.env.flat_obs_dim

    @property
    def action_dim(self):
        return self.env.action_dim

    def init_run_name(self, datetime_str=None):
        if datetime_str is None:
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log.run_name = f"{self.env.env_id}__{self.trainer.exp_name}_{datetime_str}"


@dataclass
class LSTMGates:
    i_gates: torch.Tensor
    f_gates: torch.Tensor
    g_gates: torch.Tensor
    o_gates: torch.Tensor
    c_gates: torch.Tensor
    h_gates: torch.Tensor

    @property
    def detached(self):
        return LSTMGates(
            i_gates=self.i_gates.detach(),
            f_gates=self.f_gates.detach(),
            g_gates=self.g_gates.detach(),
            o_gates=self.o_gates.detach(),
            c_gates=self.c_gates.detach(),
            h_gates=self.h_gates.detach(),
        )

    def to(self, device):
        return LSTMGates(
            i_gates=self.i_gates.to(device),
            f_gates=self.f_gates.to(device),
            g_gates=self.g_gates.to(device),
            o_gates=self.o_gates.to(device),
            c_gates=self.c_gates.to(device),
            h_gates=self.h_gates.to(device),
        )

    def transposed(self):
        """
        Returns a new LSTMGates object with all gate tensors
        transposed along the first two dimensions.
        Useful for converting between (B, T, H) and (T, B, H).
        """
        return LSTMGates(
            i_gates=self.i_gates.transpose(0, 1),
            f_gates=self.f_gates.transpose(0, 1),
            g_gates=self.g_gates.transpose(0, 1),
            o_gates=self.o_gates.transpose(0, 1),
            c_gates=self.c_gates.transpose(0, 1),
            h_gates=self.h_gates.transpose(0, 1),
        )


@dataclass
class RolloutStep:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    hxs: torch.Tensor
    cxs: torch.Tensor
    gates: LSTMGates


@dataclass
class RecurrentMiniBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_logp: torch.Tensor
    old_values: torch.Tensor
    hxs0: torch.Tensor
    cxs0: torch.Tensor
    mask: torch.Tensor  # (K, B)
    t0: int
    t1: int


@dataclass
class RecurrentBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    hxs: torch.Tensor  # (T, B, H)
    cxs: torch.Tensor  # (T, B, H)
    terminated: torch.Tensor
    truncated: torch.Tensor

    def iter_chunks(self, K: int):
        """
        Yield TBPTT chunks of length K.
        Each chunk yields:
            obs_chunk:      (K, B, obs_dim)
            actions_chunk:  (K, B, 1)
            returns_chunk:  (K, B)
            adv_chunk:      (K, B)
            old_logp_chunk: (K, B)
            old_values:     (K, B)
            hxs0:           (B, H) -- initial hidden state for this chunk
            cxs0:           (B, H)
        """
        T = self.obs.size(0)

        for t0 in range(0, T, K):
            t1 = min(t0 + K, T)

            # Hidden state at the *start* of the chunk
            hxs0 = self.hxs[t0]  # (B, H)
            cxs0 = self.cxs[t0]  # (B, H)

            mb_terminated = self.terminated[t0:t1]
            mb_truncated = self.truncated[t0:t1]
            mb_mask = 1.0 - (mb_terminated | mb_truncated).float()  # (T, B)

            yield RecurrentMiniBatch(
                obs=self.obs[t0:t1],
                actions=self.actions[t0:t1],
                returns=self.returns[t0:t1],
                advantages=self.advantages[t0:t1],
                old_logp=self.logprobs[t0:t1],
                old_values=self.values[t0:t1],
                hxs0=hxs0,
                cxs0=cxs0,
                mask=mb_mask,
                t0=t0,
                t1=t1,
            )


@dataclass
class PolicyInput:
    obs: torch.Tensor  # (N, *obs_shape)
    hxs: torch.Tensor  # (N,H)
    cxs: torch.Tensor  # (N,H)


@dataclass
class VecEnvState:
    obs: torch.Tensor  # (N, *obs_shape))
    rewards: torch.Tensor  # (N,)
    terminated: torch.Tensor  # (N,)
    truncated: torch.Tensor  # (N,)
    info: List[Any]
    hxs: torch.Tensor  # (N,H)
    cxs: torch.Tensor  # (N,H)

    @property
    def policy_input(self) -> PolicyInput:
        return PolicyInput(self.obs, self.hxs, self.cxs)

    """ Properties can't have parameters"""

    @property
    def detached_policy_input(self) -> PolicyInput:
        return PolicyInput(self.obs, self.hxs.detach(), self.cxs.detach())


@dataclass
class PolicyOutput:
    logits: torch.Tensor  # (B,A) or (B,T,A)
    values: torch.Tensor  # (B,) or (B,T)
    new_hxs: torch.Tensor  # (B,H)
    new_cxs: torch.Tensor  # (B,H)
    ar_loss: torch.Tensor  # scalar
    tar_loss: torch.Tensor  # scalar
    gates: LSTMGates

    @property
    def detached(self):
        return PolicyOutput(
            logits=self.logits.detach(),
            values=self.values.detach(),
            new_hxs=self.new_hxs.detach(),
            new_cxs=self.new_cxs.detach(),
            ar_loss=self.ar_loss.detach(),
            tar_loss=self.tar_loss.detach(),
            gates=self.gates.detached,
        )


@dataclass
class LSTMStates:
    hxs: torch.Tensor
    cxs: torch.Tensor


@dataclass
class LSTMCoreOutput:
    out: torch.Tensor
    h: torch.Tensor
    c: torch.Tensor
    ar_loss: torch.Tensor
    tar_loss: torch.Tensor
    gates: LSTMGates


@dataclass
class PolicyEvalInput:
    obs: torch.Tensor  # (N, *obs_shape)
    hxs: torch.Tensor  # (N,H)
    cxs: torch.Tensor  # (N,H)
    actions: torch.Tensor  # (N,) or (N,1)


@dataclass
class PolicyEvalOutput:
    """
    Output of the policy evaluation step for a full sequence (T, B).
    All tensors are (T, B, ...) except new_hxs/new_cxs which are (T, B, H).
    """

    values: torch.Tensor  # (T, B)
    logprobs: torch.Tensor  # (T, B)
    entropy: torch.Tensor  # (T, B)

    new_hxs: torch.Tensor  # (T, B, H)
    new_cxs: torch.Tensor  # (T, B, H)

    gates: LSTMGates  # i,f,g,o,c,h gates (T, B, H)

    ar_loss: Optional[torch.Tensor] = None
    tar_loss: Optional[torch.Tensor] = None

    def to(self, device):
        return PolicyEvalOutput(
            values=self.values.to(device),
            logprobs=self.logprobs.to(device),
            entropy=self.entropy.to(device),
            new_hxs=self.new_hxs.to(device),
            new_cxs=self.new_cxs.to(device),
            gates=self.gates.to(device),
            ar_loss=(None if self.ar_loss is None else self.ar_loss.to(device)),
            tar_loss=(None if self.tar_loss is None else self.tar_loss.to(device)),
        )

    @property
    def detached(self):
        return PolicyEvalOutput(
            values=self.values.detach(),
            logprobs=self.logprobs.detach(),
            entropy=self.entropy.detach(),
            new_hxs=self.new_hxs.detach(),
            new_cxs=self.new_cxs.detach(),
            gates=self.gates.detached,
            ar_loss=(None if self.ar_loss is None else self.ar_loss.detach()),
            tar_loss=(None if self.tar_loss is None else self.tar_loss.detach()),
        )


@dataclass
class EpisodeStats:
    episodes: int
    alive_envs: int
    max_ep_len: int
    avg_ep_len: float
    max_ep_returns: float
    avg_ep_returns: float


@dataclass
class LSTMGateSaturation:
    i_sat_low: torch.Tensor
    i_sat_high: torch.Tensor
    f_sat_low: torch.Tensor
    f_sat_high: torch.Tensor
    o_sat_low: torch.Tensor
    o_sat_high: torch.Tensor

    g_sat: torch.Tensor
    c_sat: torch.Tensor
    h_sat: torch.Tensor

    hidden_size: Optional[int] = None

    def detach(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.detach())
        return self


@dataclass
class LSTMGateEntropy:
    i_entropy: torch.Tensor
    f_entropy: torch.Tensor
    o_entropy: torch.Tensor
    g_entropy: torch.Tensor
    c_entropy: torch.Tensor
    h_entropy: torch.Tensor

    hidden_size: Optional[int] = None

    def detach(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.detach())
        return self


@dataclass
class LSTMUnitDiagnostics:
    """
    Per-unit LSTM diagnostics for research-grade interpretability.
    These metrics remain 2-D (hidden_size) and are intentionally
    kept separate from the scalar Metrics class.

    All tensors are 1-D: shape [hidden_size].
    """

    # Gate means (sigmoid gates in [0,1], g_t in [-1,1])
    i_mean: Optional[torch.Tensor] = None
    f_mean: Optional[torch.Tensor] = None
    g_mean: Optional[torch.Tensor] = None
    o_mean: Optional[torch.Tensor] = None

    # Per-unit drift (difference from previous iteration)
    i_drift: Optional[torch.Tensor] = None
    f_drift: Optional[torch.Tensor] = None
    g_drift: Optional[torch.Tensor] = None
    o_drift: Optional[torch.Tensor] = None

    saturation: Optional[LSTMGateSaturation] = None
    entropy: Optional[LSTMGateEntropy] = None

    # Hidden and cell norms per unit
    h_norm: Optional[torch.Tensor] = None
    c_norm: Optional[torch.Tensor] = None

    # Hidden and cell drift per unit
    h_drift: Optional[torch.Tensor] = None
    c_drift: Optional[torch.Tensor] = None

    # Optional: store raw hidden_size for validation
    hidden_size: Optional[int] = None

    def to(self, device):
        """Move all tensors to a device."""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def detach(self):
        """Detach all tensors from the graph."""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.detach())
        return self


@dataclass
class PolicyUpdateInfo:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    grad_norm: torch.Tensor
    policy_drift: torch.Tensor
    value_drift: torch.Tensor
    lstm_unit_diag: LSTMUnitDiagnostics


@dataclass
class Metrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    grad_norm: float = 0.0

    episodes: int = 0
    alive_envs: int = 0
    max_ep_len: int = 0
    avg_ep_len: float = 0.0
    max_ep_returns: float = 0.0
    avg_ep_returns: float = 0.0

    policy_drift: float = 0.0
    value_drift: float = 0.0

    h_norm: float = 0.0
    c_norm: float = 0.0
    h_drift: float = 0.0
    c_drift: float = 0.0

    i_mean: float = 0.0
    f_mean: float = 0.0
    g_mean: float = 0.0
    o_mean: float = 0.0

    i_drift: float = 0.0
    f_drift: float = 0.0
    g_drift: float = 0.0
    o_drift: float = 0.0

    # LSTM Gate saturation metrics
    i_sat_low: float = 0.0
    i_sat_high: float = 0.0
    f_sat_low: float = 0.0
    f_sat_high: float = 0.0
    o_sat_low: float = 0.0
    o_sat_high: float = 0.0

    g_sat: float = 0.0
    c_sat: float = 0.0
    h_sat: float = 0.0

    i_entropy: float = 0.0
    f_entropy: float = 0.0
    o_entropy: float = 0.0
    g_entropy: float = 0.0
    c_entropy: float = 0.0
    h_entropy: float = 0.0

    explained_var: float = 0.0
    kl_watchdog_triggered: int = 0
    steps: int = 0

    def accumulate(self, upd: PolicyUpdateInfo):
        self.policy_loss += upd.policy_loss.item()
        self.value_loss += upd.value_loss.item()
        self.entropy += upd.entropy.item()
        self.approx_kl += upd.approx_kl.item()
        self.clip_frac += upd.clip_frac.item()
        self.grad_norm += upd.grad_norm

        self.policy_drift += upd.policy_drift.item()
        self.value_drift += upd.value_drift.item()

        diag = upd.lstm_unit_diag
        sat = diag.saturation
        ent = diag.entropy

        # --- aggregate per-unit metrics as means ---
        if diag.h_norm is not None:
            self.h_norm += diag.h_norm.mean().item()

        if diag.c_norm is not None:
            self.c_norm += diag.c_norm.mean().item()

        if diag.h_drift is not None:
            self.h_drift += diag.h_drift.mean().item()

        if diag.c_drift is not None:
            self.c_drift += diag.c_drift.mean().item()

        # Gate means
        self.i_mean += diag.i_mean.mean().item()
        self.f_mean += diag.f_mean.mean().item()
        self.g_mean += diag.g_mean.mean().item()
        self.o_mean += diag.o_mean.mean().item()

        # Gate drift
        self.i_drift += diag.i_drift.mean().item()
        self.f_drift += diag.f_drift.mean().item()
        self.g_drift += diag.g_drift.mean().item()
        self.o_drift += diag.o_drift.mean().item()

        # Gate saturation (fractions per unit → mean over units)
        self.i_sat_low += sat.i_sat_low.mean().item()
        self.i_sat_high += sat.i_sat_high.mean().item()
        self.f_sat_low += sat.f_sat_low.mean().item()
        self.f_sat_high += sat.f_sat_high.mean().item()
        self.o_sat_low += sat.o_sat_low.mean().item()
        self.o_sat_high += sat.o_sat_high.mean().item()
        self.g_sat += sat.g_sat.mean().item()
        self.c_sat += sat.c_sat.mean().item()
        self.h_sat += sat.h_sat.mean().item()

        # Gate entropy (per unit → mean over units)
        self.i_entropy += ent.i_entropy.mean().item()
        self.f_entropy += ent.f_entropy.mean().item()
        self.o_entropy += ent.o_entropy.mean().item()
        self.g_entropy += ent.g_entropy.mean().item()
        self.c_entropy += ent.c_entropy.mean().item()
        self.h_entropy += ent.h_entropy.mean().item()

    def update_episode_stats(self, ep_stats: EpisodeStats):
        self.episodes = ep_stats.episodes
        self.alive_envs = ep_stats.alive_envs

        self.max_ep_len = ep_stats.max_ep_len
        self.avg_ep_len = ep_stats.avg_ep_len

        self.max_ep_returns = ep_stats.max_ep_returns
        self.avg_ep_returns = ep_stats.avg_ep_returns

    def normalize(self):
        if self.steps == 0:
            return

        factor = 1.0 / self.steps

        EXCLUDE = {"steps", "episodes", "alive_envs", "max_ep_len", "avg_ep_len", "max_ep_returns", "avg_ep_returns"}

        # Normalize metrics accumulated across minibatches
        for field in self.__dataclass_fields__:
            if field not in EXCLUDE:
                setattr(self, field, getattr(self, field) * factor)

    def initialize(self):
        for field in self.__dataclass_fields__:
            if field in ("steps",):
                setattr(self, field, 0)
            else:
                setattr(self, field, 0.0)

    def to_dict(self):
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__}

    def render_ppo_metrics(self, lr: float, entropy_coef: float, clip_range: float):
        # PPO metrics panel
        ppo_text = Text()

        ppo_text.append(f" Return:     {self.avg_ep_returns:.3f}\n", style="bold yellow")

        ppo_text.append(f" Length:     {self.avg_ep_len:.3f}\n", style="bold yellow")

        ppo_text.append(f" Entropy:    {self.entropy:.3f}\n", style="bold yellow")

        ppo_text.append(f" KL:         {self.approx_kl:.3f}\n", style="bold yellow")

        ppo_text.append(f" ClipFrac:   {self.clip_frac:.3f}\n", style="bold yellow")

        ppo_text.append(f" GradNorm:   {self.grad_norm:.1f}\n", style="bold yellow")

        ppo_text.append(f" ExplainedV: {self.explained_var:.3e}\n", style="bold yellow")

        ppo_text.append(f" LR:         {lr:.2e}\n", style="bold yellow")

        ppo_text.append(f" EntCoef:    {entropy_coef:.2e}\n", style="bold yellow")

        ppo_text.append(f" ClipRange:  {clip_range:.2e}\n", style="bold yellow")

        ppo_text.append(f" PolDrift:  {self.policy_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" ValDrift:  {self.value_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" h-norm:     {self.h_norm:.3f}\n", style="bold yellow")

        ppo_text.append(f" c-norm:     {self.c_norm:.3f}\n", style="bold yellow")

        ppo_text.append(f" h-drift:    {self.h_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" c-drift:    {self.c_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" i-mean:     {self.i_mean:.3f}\n", style="bold yellow")

        ppo_text.append(f" f-mean:     {self.f_mean:.3f}\n", style="bold yellow")

        ppo_text.append(f" g-mean:     {self.g_mean:.3f}\n", style="bold yellow")

        ppo_text.append(f" o-mean:     {self.o_mean:.3f}\n", style="bold yellow")

        ppo_text.append(f" i-drift:    {self.i_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" f-drift:    {self.f_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" g-drift:    {self.g_drift:.3e}\n", style="bold yellow")

        ppo_text.append(f" o-drift:    {self.o_drift:.3e}\n", style="bold yellow")

        return ppo_text

    def render_episode_stats(self, avg_ep_len_ema: float, avg_ep_returns_ema: float):
        ep_text = Text()

        ep_text.append(f" Episodes:   {self.episodes}\n", style="bold green")

        ep_text.append(f" AliveEnv:   {self.alive_envs}\n", style="bold green")

        ep_text.append(f" MaxEpLen:   {self.max_ep_len:.1f}\n", style="bold green")

        ep_text.append(f" AvgEpLen:   {self.avg_ep_len:.1f}\n", style="bold green")

        ep_text.append(f" EMA Len:    {avg_ep_len_ema:.1f}\n", style="bold green")

        ep_text.append(f" MaxReturn:  {self.max_ep_returns:.2f}\n", style="bold green")

        ep_text.append(f" AvgReturn:  {self.avg_ep_returns:.2f}\n", style="bold green")

        ep_text.append(f" EMA Return: {avg_ep_returns_ema:.2f}\n", style="bold green")

        return ep_text


@dataclass
class MetricsHistory:
    max_len: int

    ep_len: list = field(default_factory=list)
    ep_return: list = field(default_factory=list)

    kl: list = field(default_factory=list)
    entropy: list = field(default_factory=list)
    explained_var: list = field(default_factory=list)

    policy_drift: list = field(default_factory=list)
    value_drift: list = field(default_factory=list)

    h_norm: list = field(default_factory=list)
    c_norm: list = field(default_factory=list)
    h_drift: list = field(default_factory=list)
    c_drift: list = field(default_factory=list)

    i_mean: list = field(default_factory=list)
    f_mean: list = field(default_factory=list)
    g_mean: list = field(default_factory=list)
    o_mean: list = field(default_factory=list)

    i_drift: list = field(default_factory=list)
    f_drift: list = field(default_factory=list)
    g_drift: list = field(default_factory=list)
    o_drift: list = field(default_factory=list)

    i_entropy: list = field(default_factory=list)
    f_entropy: list = field(default_factory=list)
    g_entropy: list = field(default_factory=list)
    o_entropy: list = field(default_factory=list)
    c_entropy: list = field(default_factory=list)
    h_entropy: list = field(default_factory=list)

    def update(self, upd: PolicyUpdateInfo, stats: Metrics):
        self.push("kl", upd.approx_kl.item())

        diag = upd.lstm_unit_diag
        sat = diag.saturation
        ent = diag.entropy

        mapping = {
            "i_mean": diag.i_mean.mean(),
            "f_mean": diag.f_mean.mean(),
            "g_mean": diag.g_mean.mean(),
            "o_mean": diag.o_mean.mean(),
            "i_drift": diag.i_drift.mean(),
            "f_drift": diag.f_drift.mean(),
            "g_drift": diag.g_drift.mean(),
            "o_drift": diag.o_drift.mean(),
            "h_norm": (diag.h_norm.mean() if diag.h_norm is not None else stats.h_norm),
            "c_norm": (diag.c_norm.mean() if diag.c_norm is not None else stats.c_norm),
            "h_drift": (diag.h_drift.mean() if diag.h_drift is not None else stats.h_drift),
            "c_drift": (diag.c_drift.mean() if diag.c_drift is not None else stats.c_drift),
            "i_entropy": ent.i_entropy.mean(),
            "f_entropy": ent.f_entropy.mean(),
            "g_entropy": ent.g_entropy.mean(),
            "o_entropy": ent.o_entropy.mean(),
            "c_entropy": ent.c_entropy.mean(),
            "h_entropy": ent.h_entropy.mean(),
            "i_sat_low": sat.i_sat_low.mean(),
            "i_sat_high": sat.i_sat_high.mean(),
            "f_sat_low": sat.f_sat_low.mean(),
            "f_sat_high": sat.f_sat_high.mean(),
            "o_sat_low": sat.o_sat_low.mean(),
            "o_sat_high": sat.o_sat_high.mean(),
            "g_sat": sat.g_sat.mean(),
            "c_sat": sat.c_sat.mean(),
            "h_sat": sat.h_sat.mean(),
            "explained_var": stats.explained_var,
            "ep_return": stats.avg_ep_returns,
            "ep_len": stats.avg_ep_len,
        }

        for name, tensor in mapping.items():
            self.push(name, tensor.item())

    def push(self, name: str, value: float):
        hist = getattr(self, name)
        hist.append(value)

        if len(hist) > self.max_len:
            hist.pop(0)

    def render_ppo_history(self, ppo_text: Text):
        ppo_text.append("\n Return Trend: ", style="bold cyan")

        ppo_text.append(sparkline(self.ep_return), style="cyan")

        ppo_text.append("\n KL Trend:    ", style="bold cyan")
        ppo_text.append(sparkline(self.kl, width=20), style="cyan")

        ppo_text.append("\n Entropy Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.entropy, width=20), style="magenta")

        ppo_text.append("\n EV Trend:    ", style="bold cyan")
        ppo_text.append(sparkline(self.explained_var, width=20), style="green")

        ppo_text.append("\n PolDrift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.policy_drift), style="cyan")

        ppo_text.append("\n ValDrift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.value_drift), style="green")

        ppo_text.append("\n h-norm Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.h_norm), style="cyan")

        ppo_text.append("\n c-norm Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.c_norm), style="cyan")

        ppo_text.append("\n h-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.h_drift), style="green")

        ppo_text.append("\n c-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.c_drift), style="green")

        ppo_text.append("\n i-mean Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.i_mean), style="cyan")
        ppo_text.append("\n f-mean Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.f_mean), style="cyan")

        ppo_text.append("\n g-mean Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.g_mean), style="cyan")

        ppo_text.append("\n o-mean Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.o_mean), style="cyan")

        ppo_text.append("\n i-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.i_drift), style="green")

        ppo_text.append("\n f-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.f_drift), style="green")

        ppo_text.append("\n g-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.g_drift), style="green")

        ppo_text.append("\n o-drift Tr.: ", style="bold cyan")
        ppo_text.append(sparkline(self.o_drift), style="green")

    def render_episode_history(self, ep_text: Text):
        ep_text.append("\n Length Trend: ", style="bold cyan")

        ep_text.append(sparkline(self.ep_len), style="cyan")


def initialize_config(cfg: Config, **kwargs):
    # Set torch flags
    torch.backends.cudnn.deterministic = cfg.trainer.torch_deterministic

    if cfg.trainer.debug_mode:
        torch.autograd.set_detect_anomaly(True)

    # Build dummy env
    dummy_env = gym.make(cfg.env.env_id)

    cfg.env.obs_space = dummy_env.observation_space
    cfg.env.action_dim = dummy_env.action_space.n
    cfg.env.max_episode_steps = dummy_env.spec.max_episode_steps

    cfg.env.flat_obs_dim = get_flat_obs_dim(cfg.env.obs_space)

    dummy_env.close()

    # Build run name
    cfg.init_run_name(kwargs.get("datetime_str", None))

    return cfg


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
