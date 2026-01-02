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
    learning_rate: float = 3e-4
    """Learning rate"""
    clip_range: float = 0.2
    """ Clip coefficient"""
    update_epochs: int = 4
    """ Number of update eophics"""
    target_kl: float = 0.01
    """ Target KL divergence threshold"""
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
    values: torch.Tensor       # (N,)
    logprobs: torch.Tensor     # (N,)
    entropy: torch.Tensor      # scalar
    new_hxs: torch.Tensor      # (N,H)
    new_cxs: torch.Tensor      # (N,H)
    ar_loss: torch.Tensor      # scalar
    tar_loss: torch.Tensor     # scalar