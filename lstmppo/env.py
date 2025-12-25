import gymnasium as gym
import popgym
from gymnasium.vector import SyncVectorEnv


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        return env
    return thunk

def make_vector_env(cfg):

    venv = SyncVectorEnv([make_env(cfg.env_id) for _ in range(cfg.num_envs)])

    return venv

class RecurrentRolloutBuffer:
    def __init__(self, T, N, obs_shape, action_dim, hidden_size, device):
        self.T = T
        self.N = N
        self.device = device

        self.obs = torch.zeros(T, N, *obs_shape, device=device)
        self.actions = torch.zeros(T, N, 1, device=device)  # discrete
        self.rewards = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, N, device=device)
        self.logprobs = torch.zeros(T, N, device=device)

        self.terminated = torch.zeros(T, N, device=device, dtype=torch.bool)
        self.truncated = torch.zeros(T, N, device=device, dtype=torch.bool)

        # Hidden states at start of each timestep
        self.hxs = torch.zeros(T, N, hidden_size, device=device)
        self.cxs = torch.zeros(T, N, hidden_size, device=device)

        self.step = 0

    def add(self, obs, actions, rewards, values, logprobs,
            terminated, truncated, hxs, cxs):
        t = self.step
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions.unsqueeze(-1))  # (N,) -> (N,1)
        self.rewards[t].copy_(rewards)
        self.values[t].copy_(values)
        self.logprobs[t].copy_(logprobs)
        self.terminated[t].copy_(terminated)
        self.truncated[t].copy_(truncated)
        self.hxs[t].copy_(hxs)
        self.cxs[t].copy_(cxs)
        self.step += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        T, N = self.T, self.N
        advantages = torch.zeros(T, N, device=self.device)
        last_gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            true_terminal = self.terminated[t]
            bootstrap = ~true_terminal  # truncated episodes still bootstrap

            next_value = last_value if t == T - 1 else self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * bootstrap
                - self.values[t]
            )

            last_gae = delta + gamma * lam * last_gae * bootstrap
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = self.values + advantages

    def get_recurrent_minibatches(self, batch_envs):
        """
        Entire sequences per environment:
        yields dict with tensors shaped (T, B, ...)
        """
        N = self.N
        env_indices = torch.randperm(N, device=self.device)

        for start in range(0, N, batch_envs):
            idx = env_indices[start:start + batch_envs]

            yield {
                "obs": self.obs[:, idx],             # (T, B, obs_dim)
                "actions": self.actions[:, idx],     # (T, B, 1)
                "values": self.values[:, idx],
                "logprobs": self.logprobs[:, idx],
                "returns": self.returns[:, idx],
                "advantages": self.advantages[:, idx],
                "hxs": self.hxs[0, idx],             # (B, hidden)
                "cxs": self.cxs[0, idx],             # (B, hidden)
                "terminated": self.terminated[:, idx],
                "truncated": self.truncated[:, idx],
            }

    def reset(self):
        self.step = 0
