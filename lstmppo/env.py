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
    dummy_env = gym.make(cfg.env_id)
    obs_shape = dummy_env.observation_space.shape
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    return obs_shape, action_dim, venv
