from dataclasses import dataclass
import tyro
import time
import torch
import gymnasium as gym
import popgym
from .types import PPOConfig


def initialize(seconds_since_epoch=None):
    
    cfg = tyro.cli(PPOConfig)

    if seconds_since_epoch is None:
        seconds_since_epoch = int(time.time())

    cfg.run_name =\
        f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__" +\
        f"{seconds_since_epoch}"

    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    cfg.device = torch.device("cuda" if torch.cuda.is_available()
                               and cfg.cuda else "cpu")
    
    dummy_env = gym.make(cfg.env_id)
    cfg.obs_shape = dummy_env.observation_space.shape
    cfg.action_dim = dummy_env.action_space.n
    dummy_env.close()

    return cfg
