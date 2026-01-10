from numpy.random import Generator, MT19937, SeedSequence
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch

from .types import Config, VecEnvState, PolicyInput, LSTMStates, EpisodeStats
from .obs_encoder import flatten_obs


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        return env
    return thunk


class RecurrentVecEnvWrapper:
    """
    Wrap vectorized env and manage per-env LSTM hidden states.
    """

    def __init__(self,
                 cfg: Config,
                 device):

        self.venv = SyncVectorEnv([make_env(cfg.env.env_id)
                                   for _ in range(cfg.env.num_envs)])

        self.num_envs = cfg.env.num_envs
        self.hidden_size = cfg.lstm.lstm_hidden_size
        self.device = device

        self.hxs = torch.zeros(self.num_envs,
                               self.hidden_size,
                               device=self.device)

        self.cxs = torch.zeros(self.num_envs,
                               self.hidden_size,
                               device=self.device)
        
        self.last_terminated = torch.zeros(self.num_envs,
                                           dtype=torch.bool,
                                           device=self.device)

        self.ep_len = torch.zeros(self.num_envs,
                                  dtype=torch.int32,
                                  device=self.device)
        
        self.ep_return = torch.zeros(self.num_envs,
                                     dtype=torch.float32,
                                     device=self.device)

        self.completed_ep_returns = []

        self.completed_ep_lens = []   # store results for logging
        
    @property
    def observation_space(self):
        return self.venv.single_observation_space
    
    @property
    def hidden_state(self):
        return LSTMStates(self.hxs, self.cxs)

    def reset(self, seed=None) -> VecEnvState:

        rng = Generator(MT19937(SeedSequence(seed)))
        seeds = [int(elem * 1E9) for elem in rng.random(self.num_envs)]

        obs, info = self.venv.reset(seed=seeds)

        obs = flatten_obs(obs,
                          self.venv.single_observation_space)

        self.hxs.zero_()
        self.cxs.zero_()
        self.last_terminated.zero_()

        obs = torch.as_tensor(obs,
                              device=self.device,
                              dtype=torch.float32)

        return VecEnvState(
            obs=obs,
            rewards=torch.zeros(self.num_envs,
                                device=self.device),
            terminated=torch.zeros(self.num_envs,
                                   dtype=torch.bool,
                                   device=self.device),
            truncated=torch.zeros(self.num_envs,
                                  dtype=torch.bool,
                                  device=self.device),
            info=info,
            hxs=self.hxs.clone(),
            cxs=self.cxs.clone(),
        )

    def set_initial_lstm_states(self,
                                last_lstm_states: LSTMStates) -> None:
        """
        Called at the start of a new rollout.`
        hxs, cxs: (num_envs, hidden_size)
        Only applied to environments that did NOT terminate.
        """

        if last_lstm_states.hxs is None:
            # first rollout ever
            initial_hxs = torch.zeros(self.num_envs,
                                      self.hidden_size,
                                      device=self.device)

            initial_cxs = torch.zeros(self.num_envs,
                                      self.hidden_size,
                                      device=self.device)
        else:
            initial_hxs = last_lstm_states.hxs
            initial_cxs = last_lstm_states.cxs

        # Only carry over states for environments that didn't terminate or
        # truncate
        carry_mask = ~self.last_terminated 

        self.hxs[carry_mask] = initial_hxs[carry_mask]
        self.cxs[carry_mask] = initial_cxs[carry_mask]

    def step(self, actions: torch.Tensor) -> VecEnvState:

        """
        actions: (N,) or (N,1) tensor, discrete.
        """
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).cpu().numpy()
        else:
            actions_np = actions.cpu().numpy()

        obs, rewards, terminated, truncated, info =\
            self.venv.step(actions_np)
        
        obs = flatten_obs(obs,
                          self.venv.single_observation_space)

        # ------ Convert flags to tensors ------
        terminated = torch.as_tensor(terminated,
                                     device=self.device,
                                     dtype=torch.bool)
        
        truncated = torch.as_tensor(truncated,
                                    device=self.device,
                                    dtype=torch.bool)
        
        # increment all alive envs
        self.ep_len += 1

        # Keep track of episode rewards
        self.ep_return += torch.as_tensor(rewards,
                                          device=self.device,
                                          dtype=torch.float32)

        """
        Vectorized environment wrapper is the only place that knows which
        environments:

        • 	Terminated at this step
        • 	Truncated due to time limit
        • 	Are still alive
        • 	Have just been reset

        The trainer and buffer only see batched tensors — they don’t 
        know which envs just died.
        """
        done_mask = terminated | truncated

        if done_mask.any():

            self.hxs[done_mask].zero_()
            self.cxs[done_mask].zero_()

            # record completed episode lengths
            finished_lengths = self.ep_len[done_mask].cpu().tolist()
            self.completed_ep_lens.extend(finished_lengths)

            # record completed episode rewards
            finished_returns = self.ep_return[done_mask].cpu().tolist()
            self.completed_ep_returns.extend(finished_returns)
            self.ep_return[done_mask] = 0

            # reset counters for those envs
            self.ep_len[done_mask] = 0

        obs = torch.as_tensor(obs,
                              device=self.device,
                              dtype=torch.float32)
        
        rewards = torch.as_tensor(rewards,
                                  device=self.device,
                                  dtype=torch.float32)

        self.last_terminated = done_mask.clone()

        return VecEnvState(
            obs=obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
            hxs=self.hxs.clone(),
            cxs=self.cxs.clone(),
        )

    def update_hidden_states(self,
                             new_hxs: torch.Tensor,
                             new_cxs: torch.Tensor) -> None:

        self.hxs.copy_(new_hxs)
        self.cxs.copy_(new_cxs)

    def get_episode_stats(self):

        episodes = len(self.completed_ep_lens)

        if episodes == 0:
            
            max_ep_len = 0
            avg_ep_len = 0.0
            max_ep_returns = 0.0
            avg_ep_returns = 0.0
        else:
            max_ep_len = max(self.completed_ep_lens)
            avg_ep_len = sum(self.completed_ep_lens) / episodes
            
            max_ep_returns = max(self.completed_ep_returns)

            avg_ep_returns =\
                (sum(self.completed_ep_returns) /
                 len(self.completed_ep_returns))

        self.completed_ep_lens.clear()
        self.completed_ep_returns.clear()

        return EpisodeStats(
            episodes=episodes,
            alive_envs=(self.ep_len > 0).sum().item(),
            max_ep_len=max_ep_len,
            avg_ep_len=float(avg_ep_len),
            max_ep_returns=max_ep_returns,
            avg_ep_returns=float(avg_ep_returns)
        )


# Helper on VecEnvState side
def to_policy_input(env_state: VecEnvState) -> PolicyInput:
    return PolicyInput(
        obs=env_state.obs,
        hxs=env_state.hxs,
        cxs=env_state.cxs,
    )