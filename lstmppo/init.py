from dataclasses import dataclass
from typing import Optional
import numpy
import tyro
import time
import torch
import wandb
import random
import numpy as np


@dataclass
class Args:
    exp_name: str = "RLTestDrive"
    """the name of this experiment"""
    seed: int = 351530767
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    horizon: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 16
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
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
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def initialize(seconds_since_epoch=None):
    
    params = tyro.cli(Args)
    params.batch_size = int(params.num_envs * params.horizon)
    params.minibatch_size = int(params.batch_size // params.num_minibatches)
    params.num_iterations = params.total_timesteps // params.batch_size

    if seconds_since_epoch is None:
        seconds_since_epoch = int(time.time())

    params.run_name =\
        f"{params.env_id}__{params.exp_name}__{params.seed}__" +\
        f"{seconds_since_epoch}"

    if params.track:

        wandb.init(
            project=params.wandb_project_name,
            entity=params.wandb_entity,
            sync_tensorboard=True,
            config=vars(params),
            name=params.run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    params.device = torch.device("cuda" if torch.cuda.is_available()
                                 and params.cuda else "cpu")

    return params
