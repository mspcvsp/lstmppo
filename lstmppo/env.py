from numpy.random import Generator, MT19937, SeedSequence
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch

from .types import Config, VecEnvState, PolicyInput, LSTMStates


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

    def reset(self, seed=None) -> VecEnvState:

        rng = Generator(MT19937(SeedSequence(seed)))
        seeds = [int(elem * 1E9) for elem in rng.random(self.num_envs)]

        obs, info = self.venv.reset(seed=seeds)
        
        # flatten dict/tuple observations
        obs = self._encode_obs(obs)

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
        
        obs = self._encode_obs(obs)

        # ------ Convert flags to tensors ------
        terminated = torch.as_tensor(terminated,
                                     device=self.device,
                                     dtype=torch.bool)
        
        truncated = torch.as_tensor(truncated,
                                    device=self.device,
                                    dtype=torch.bool)

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

    def _encode_obs(self,
                    obs):

        if isinstance(obs, dict):

            return {k: torch.as_tensor(v,
                                       device=self.device,
                                       dtype=torch.float32)
                    for k, v in obs.items()}
        # ------------------------------------------
        elif isinstance(obs, tuple):

            return tuple(torch.as_tensor(v,
                                         device=self.device,
                                         dtype=torch.float32)
                        for v in obs)
        # ------------------------------------------
        else:

            return torch.as_tensor(obs,
                                   device=self.device,
                                   dtype=torch.float32)

    def update_hidden_states(self,
                             new_hxs: torch.Tensor,
                             new_cxs: torch.Tensor) -> None:

        self.hxs.copy_(new_hxs)
        self.cxs.copy_(new_cxs)


# Helper on VecEnvState side
def to_policy_input(env_state: VecEnvState) -> PolicyInput:
    return PolicyInput(
        obs=env_state.obs,
        hxs=env_state.hxs,
        cxs=env_state.cxs,
    )