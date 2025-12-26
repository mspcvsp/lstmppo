from dataclasses import dataclass
import tyro
import time
import torch
import random
import numpy as np
import gymnasium as gym
import popgym


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
    rollout_steps: int = 128
    """ Horizon"""
    gamma: float = 0.99
    """ Discount factor"""
    gae_lambda: float = 0.95
    """ Generalized Advantage Estimate (GAE) lambda"""
    learning_rate: float = 3e-4
    """Learning rate"""
    clip_coef: float = 0.2
    """ Clip coefficient"""
    update_epochs: int = 4
    """ Number of update eophics"""
    batch_envs: int = 4 
    """number of envs per recurrent minibatch"""
    ent_coef: float = 0.01
    """Fixed coefficient of the entropy if annealing is disabled"""
    anneal_entropy_flag: bool = True
    """Toggle entropy coefficient annealing"""
    start_ent_coef: float = 0.01
    """Starting value of entropy coefficient for annealing"""
    end_ent_coef: float = 0.0
    """Ending value of entropy coefficient for annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    hidden_size: int = 64
    """LSTM hidden size"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """the name of this experiment"""
    seed: int = 351530767
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""


def initialize(seconds_since_epoch=None):
    
    cfg = tyro.cli(PPOConfig)

    if seconds_since_epoch is None:
        seconds_since_epoch = int(time.time())

    cfg.run_name =\
        f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__" +\
        f"{seconds_since_epoch}"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    cfg.device = torch.device("cuda" if torch.cuda.is_available()
                               and cfg.cuda else "cpu")
    
    dummy_env = gym.make(cfg.env_id)
    cfg.obs_shape = dummy_env.observation_space.shape
    cfg.action_dim = dummy_env.action_space.n
    dummy_env.close()

    return cfg
