"""
• 	Config dataclasses must be pure data containers
• 	No side effects in __init__
• 	No environment creation
• 	No torch backend mutation
"""
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, List
import torch
import gymnasium as gym
import popgym
from .types import build_obs_encoder
from .obs_encoder import get_flat_obs_dim


@dataclass
class PPOHyperparams:

    gamma: float = 0.99
    """ Discount factor"""
    gae_lambda: float = 0.95
    """ Generalized Advantage Estimate (GAE) lambda"""
    initial_clip_range: float = 0.2
    """ Initial clip range """
    update_epochs: int = 4
    """ Number of update epochs """
    vf_coef: float = 0.5
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
    lstm_ar_coef: float = 2.0
    """LSTM activation regularization (AR)"""
    lstm_tar_coef: float = 1.0
    """LSTM temporal activation regularization (TAR)"""


@dataclass
class ScheduleConfig:

    base_lr: float = 3e-4
    """ Base learning rate """
    lr_warmup_pct: float = 5.0
    """ Learning rate warmup percentage """
    lr_final_pct: float = 10.0
    """ Final learning rate percentage of base learning rate """
    anneal_entropy_flag: bool = True
    """ Toggle entropy coefficient annealing """
    start_entropy_coef: float = 0.1
    """ Starting value of entropy coefficient for annealing """
    end_entropy_coef: float = 0.0


@dataclass
class TrainerConfig:

    cuda: bool = True
    """ Whether to use CUDA """
    torch_deterministic: bool = True
    """ Sets the value of torch.backends.cudnn.deterministic """
    rollout_steps: int = 128
    """ Horizon"""
    update_epochs: int = 4
    """ Number of optimize policy update epochs """
    updates_per_checkpoint: int = 10
    """ Number of updates / checkpoint """
    debug_mode: bool = True
    """ Toggles debug mode """
    seed: int = 351530767
    """seed of the experiment"""
    tbptt_chunk_len: int = 16
    """ Truncated BPTT steps """
    exp_name: str = "RLWarmup"
    """ Experiment name """


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

    env_id: str = "popgym-PositionOnlyCartPoleEasy-v0"
    """Environment identifier"""
    num_envs: int = 16
    """ Number of environments """
    obs_space: gym.Space | None = None
    """ Observation space """
    flat_obs_dim: int
    """ Flattened observation dimension """
    action_dim: int = 0
    """ Action dimension """
    max_episode_steps: int = None
    """ Maximum number of steps per episode """


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
            rollout_steps = self.trainer.rollout_steps,
            num_envs = self.env.num_envs,
            mini_batch_envs = self.trainer.mini_batch_envs,
            gamma=self.ppo.gamma,
            lam=self.ppo.gae_lambda,
            lstm_hidden_size=self.lstm.lstm_hidden_size
        )

    def init_run_name(self,
                      datetime_str=None):

        if datetime_str is None:
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log.run_name =\
            f"{self.env.env_id}__{self.trainer.exp_name}_{datetime_str}"


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
        B = self.obs.size(1)

        for t0 in range(0, T, K):
            t1 = min(t0 + K, T)

            # Hidden state at the *start* of the chunk
            hxs0 = self.hxs[t0]   # (B, H)
            cxs0 = self.cxs[t0]   # (B, H)

            mb_terminated = self.terminated[t0:t1]
            mb_truncated = self.truncated[t0:t1]
            mb_mask = 1.0 - (mb_terminated | mb_truncated).float() # (T, B)

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
                t1=t1
            )

@dataclass
class PolicyInput:
    obs: torch.Tensor # (N, *obs_shape)
    hxs: torch.Tensor # (N,H)
    cxs: torch.Tensor # (N,H)


@dataclass
class VecEnvState:
    obs: torch.Tensor        # (N, *obs_shape))
    rewards: torch.Tensor    # (N,)
    terminated: torch.Tensor # (N,)
    truncated: torch.Tensor  # (N,)
    info: List[Any]
    hxs: torch.Tensor       # (N,H)
    cxs: torch.Tensor       # (N,H)

    @property
    def policy_input(self,
                     detach: bool = False) -> PolicyInput:

        if detach:
            return PolicyInput(
                obs=self.obs,
                hxs=self.hxs.detach(),
                cxs=self.cxs.detach(),
            )
        
        return PolicyInput(self.obs,
                              self.hxs,
                              self.cxs)
    

@dataclass
class PolicyOutput:
    logits: torch.Tensor       # (B,A) or (B,T,A)
    values: torch.Tensor       # (B,) or (B,T)
    new_hxs: torch.Tensor      # (B,H)
    new_cxs: torch.Tensor      # (B,H)
    ar_loss: torch.Tensor      # scalar
    tar_loss: torch.Tensor     # scalar


@dataclass
class LSTMStates:
    hxs: torch.Tensor
    cxs: torch.Tensor


@dataclass
class PolicyEvalInput:
    obs: torch.Tensor      # (N, *obs_shape)
    hxs: torch.Tensor      # (N,H)
    cxs: torch.Tensor      # (N,H)
    actions: torch.Tensor  # (N,) or (N,1)


@dataclass
class PolicyEvalOutput:
    values: torch.Tensor    # (T, B)
    logprobs: torch.Tensor  # (T, B)
    entropy: torch.Tensor   # (T, B)
    new_hxs: torch.Tensor   # (B,H)
    new_cxs: torch.Tensor   # (B,H)
    ar_loss: torch.Tensor   # scalar
    tar_loss: torch.Tensor  # scalar


@dataclass
class PolicyUpdateInfo:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    grad_norm: float


def initialize_config(cfg: Config,
                      **kwargs):

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