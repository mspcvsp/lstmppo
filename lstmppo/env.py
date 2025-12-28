import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch
from .types import PPOConfig, VecEnvState, PolicyInput, LSTMStates


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
                 cfg: PPOConfig,
                 venv: SyncVectorEnv):

        self.venv = venv
        self.num_envs = venv.num_envs
        self.hidden_size = cfg.lstm_hidden_size
        self.device = cfg.device

        self.hxs = torch.zeros(self.num_envs,
                               self.hidden_size,
                               device=cfg.device)

        self.cxs = torch.zeros(self.num_envs,
                               self.hidden_size,
                               device=cfg.device)
        
        self.last_terminated = torch.zeros(self.num_envs,
                                           dtype=torch.bool,
                                           device=self.device)

    def reset(self) -> VecEnvState:

        obs, info = self.venv.reset()

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

        # Only carry over states for non-terminal envs
        carry_mask = ~self.last_terminated  # stored from previous rollout

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

        obs, rewards, terminated, truncated, info = self.venv.step(actions_np)

        obs = torch.as_tensor(obs,
                              device=self.device,
                              dtype=torch.float32)
        
        rewards = torch.as_tensor(rewards,
                                  device=self.device,
                                  dtype=torch.float32)

        terminated = torch.as_tensor(terminated,
                                     device=self.device,
                                     dtype=torch.bool)
        
        truncated = torch.as_tensor(truncated,
                                    device=self.device,
                                    dtype=torch.bool)
        
        self.last_terminated = terminated.clone()

        # Reset hidden states only for true terminals
        done_mask = terminated

        if done_mask.any():

            self.hxs[done_mask].zero_()
            self.cxs[done_mask].zero_()

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


# Helper on VecEnvState side
def to_policy_input(env_state: VecEnvState) -> PolicyInput:
    return PolicyInput(
        obs=env_state.obs,
        hxs=env_state.hxs,
        cxs=env_state.cxs,
    )