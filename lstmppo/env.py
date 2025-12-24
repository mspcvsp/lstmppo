import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch.utils.tensorboard import SummaryWriter
from .agent import Agent
import numpy as np
import pdb


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def setup_env(params):

    envs =\
        SubprocVecEnv([make_env(params.env_id,
                                i,
                                params.capture_video,
                                params.run_name) for i in
                                range(params.num_envs)],)
    
    return envs


class Environment:

    def __init__(self,
                 params,
                 **kwargs):

        self.device = params.device
        self.num_steps = params.num_steps
        self.num_envs = params.num_envs
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda

        self.next_obs = None
        self.next_done = None
        self.advantages = None
        self.returns = None
        self.global_step = 0

        self.envs = setup_env(params)
        self.envs.seed(kwargs.get("seed", params.seed))

        obs_space_dims = self.envs.observation_space.shape
        action_space_dims = self.envs.action_space.shape

        self.obs = torch.zeros((params.num_steps,
                                params.num_envs) +
                                obs_space_dims).to(params.device)

        self.actions = torch.zeros((params.num_steps,
                                    params.num_envs) +
                                    action_space_dims).to(params.device)

        self.logprobs = torch.zeros((params.num_steps,
                                     params.num_envs)).to(params.device)
        
        self.rewards = torch.zeros((params.num_steps,
                                    params.num_envs)).to(params.device)
        
        self.dones = torch.zeros((params.num_steps,
                                  params.num_envs)).to(params.device)
        
        self.values = torch.zeros((params.num_steps,
                                   params.num_envs)).to(params.device)
        
        self.writer = SummaryWriter(f"runs/{params.run_name}")

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                     for key, value in
                                                         vars(params).items()])),
            )

        self.reset()
        
    def init_agent(self):

         return Agent(self.envs).to(self.device)
    
    def get_obs_space_dims(self):

        return self.envs.observation_space.shape

    def get_action_space_dims(self):

        return self.envs.action_space.shape

    def reset(self):

        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.num_envs).to(self.device)
        self.global_step = 0

    def rollout(self,
                agent,
                **kwargs):

        episodic_reward_list = []
        episodic_length_list = []

        num_steps = kwargs.get("num_steps",
                               self.num_steps)
        
        base_key = "episode"

        for step in range(0, num_steps):

            self.global_step += self.num_envs
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():

                action, logprob, _, value =\
                    agent.get_action_and_value(self.next_obs)

                self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

            self.next_obs, reward, self.next_done, infos =\
                self.envs.step(action.cpu().numpy())

            self.next_obs = torch.Tensor(self.next_obs).to(self.device)
            self.next_done = torch.Tensor(self.next_done).to(self.device)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)

            for env_info in infos:
                if base_key in env_info:
                    episodic_reward_list.append(env_info[base_key]["r"])
                    episodic_length_list.append(env_info[base_key]["l"])

        if len(episodic_reward_list) > 0:

            avg_episodic_reward = np.mean(episodic_reward_list)
            avg_episodic_length = int(np.mean(episodic_length_list) + 0.5)

            self.writer.add_scalar("charts/avg_episodic_reward",
                                avg_episodic_reward,
                                self.global_step)
            
            self.writer.add_scalar("charts/avg_episodic_length",
                                avg_episodic_length,
                                self.global_step)

            status_str =\
                f"global_step={self.global_step}, " +\
                f"avg_episodic_reward={avg_episodic_reward:.2f}, " +\
                f"avg_episodic_length={avg_episodic_length}"

            print(status_str)

        """
        Generalized Advantage Estimation
        """
        with torch.no_grad():

            next_value = agent.get_value(self.next_obs).reshape(1, -1)
            self.advantages = torch.zeros_like(self.rewards).to(self.device)

            lastgaelam = 0
            for t in reversed(range(self.num_steps)):

                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]

                delta =\
                    self.rewards[t] +\
                    self.gamma * nextvalues * nextnonterminal -\
                    self.values[t]

                self.advantages[t] = lastgaelam =\
                    delta +\
                    self.gamma * self.gae_lambda *\
                    nextnonterminal * lastgaelam

            self.returns = self.advantages + self.values

    def add_tb_scalar(self,
                      tag,
                      scalar_value):

        self.writer.add_scalar(tag,
                               scalar_value,
                               self.global_step)