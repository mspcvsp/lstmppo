from dataclasses import dataclass
from typing import Any, List
import torch


@dataclass
class PPOConfig:
    cuda: bool = True
    """GPU/CPU toggle"""
    env_id: str = "popgym-PositionOnlyCartPoleEasy-v0"
    """Environment identifier"""
    exp_name: str = "RLWarmup"
    """Experiment name"""
    total_steps: int = 200_000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """ Number of environments """
    mini_batch_envs: int = 4 
    """number of envs per recurrent minibatch"""
    rollout_steps: int = 128
    """ Horizon"""
    gamma: float = 0.99
    """ Discount factor"""
    gae_lambda: float = 0.95
    """ Generalized Advantage Estimate (GAE) lambda"""
    base_lr: float = 3e-4
    """Base learning rate"""
    clip_range: float = 0.2
    """ Clip coefficient"""
    update_epochs: int = 4
    """ Number of update eophics"""
    fixed_entropy_coef: float = 0.01
    """Fixed coefficient of the entropy if annealing is disabled"""
    anneal_entropy_flag: bool = True
    """Toggle entropy coefficient annealing"""
    start_entropy_coef: float = 0.01
    """Starting value of entropy coefficient for annealing"""
    end_entropy_coef: float = 0.0
    """Ending value of entropy coefficient for annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """the name of this experiment"""
    seed: int = 351530767
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
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
    tbptt_steps: int = 16
    """ Truncated BPTT steps """
    tb_logdir: str = "./tb_logs"
    """ TensorBoard log directory """
    jsonl_path: str = "./jsonl"
    """ JSONL (one JSON object per line) log directory """
    target_kl: float = 0.005
    """ Target KL divergence threshold """
    perc_warmup_updates: float = 5.0
    """ Percentage of warmup updates """
    end_lr_perc: float = 10.0
    """ Ending learning rate percentage of base learning rate"""

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

    def to_policy_input(self,
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
