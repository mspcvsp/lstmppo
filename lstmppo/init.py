from dataclasses import dataclass
import tyro
from datetime import datetime
import torch
import gymnasium as gym
import popgym
from .types import PPOConfig


def initialize(datetime_str = None):
    
    cfg = tyro.cli(PPOConfig)

    if datetime_str is None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg.run_name = f"{cfg.env_id}__{cfg.exp_name}_{datetime_str}"

    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    if cfg.debug_mode:
        torch.autograd.set_detect_anomaly(True)

    cfg.device = torch.device("cuda" if torch.cuda.is_available()
                               and cfg.cuda else "cpu")
    
    dummy_env = gym.make(cfg.env_id)
    cfg.obs_shape = dummy_env.observation_space.shape
    cfg.action_dim = dummy_env.action_space.n
    dummy_env.close()

    return cfg
