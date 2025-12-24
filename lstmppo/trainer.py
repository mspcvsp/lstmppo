import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .env import Environment


class PPOTrainer(object):

    def __init__(self,
                 params):
    
        self.b_obs = None
        self.b_actions = None
        self.b_logprobs = None
        self.b_advantages = None
        self.b_returns = None
        self.b_values = None

        self.learning_rate = params.learning_rate
        self.anneal_lr = params.anneal_lr 
        self.num_iterations = params.num_iterations
        self.batch_size = params.batch_size
        self.minibatch_size = params.minibatch_size
        self.clip_coef = params.clip_coef
        self.norm_adv = params.norm_adv
        self.vf_coef = params.vf_coef
        self.clip_vloss = params.clip_vloss
        self.max_grad_norm = params.max_grad_norm
        self.update_epochs = params.update_epochs

        self.ent_coef = params.ent_coef
        self.anneal_entropy_flag = params.anneal_entropy_flag
        self.start_ent_coef = params.start_ent_coef
        self.end_ent_coef = params.end_ent_coef
  
        self.total_n_mb =\
            self.update_epochs *\
            self.batch_size // self.minibatch_size

        self.mb_number = None
        self.mb_clipfracs = None
        self.mb_old_approx_kl = None
        self.mb_approx_kl = None
        self.mb_pg_loss = None
        self.mb_v_loss = None
        self.mb_entropy_loss = None
        self.mb_loss = None

        self.batch_idx = np.arange(self.batch_size)

        self.vec_env = Environment(params)
        self.agent = self.vec_env.init_agent()

        self.optimizer = optim.Adam(self.agent.parameters(),
                                    lr=self.learning_rate,
                                    eps=1e-5)
        
    def rollout(self,
                **kwargs):

        self.vec_env.rollout(self.agent,
                             **kwargs)
        
        self.b_obs = self.vec_env.obs.reshape((-1,) +
                                         self.vec_env.get_obs_space_dims())
        
        self.b_actions =\
            self.vec_env.actions.reshape((-1,) +
                                         self.vec_env.get_action_space_dims())

        self.b_logprobs = self.vec_env.logprobs.reshape(-1)
        self.b_advantages = self.vec_env.advantages.reshape(-1)
        self.b_returns = self.vec_env.returns.reshape(-1)
        self.b_values = self.vec_env.values.reshape(-1)

    def __call__(self,
                 **kwargs):

        num_iterations = kwargs.get("num_iterations",self.num_iterations)

        if num_iterations > 0:
            anneal_lr_flag = self.anneal_lr
        else:
            anneal_lr_flag = False

        for iteration in range(1, num_iterations + 1):

            if anneal_lr_flag:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            if self.anneal_entropy_flag:
                frac = iteration / self.num_iterations

                self.ent_coef =\
                    self.start_ent_coef +\
                    frac * (self.end_ent_coef - self.start_ent_coef)

            self.rollout()

            self.init_training_epoch()

            for _ in range(self.update_epochs):

                np.random.shuffle(self.batch_idx)

                for start in range(0, self.batch_size, self.minibatch_size):

                    end = start + self.minibatch_size
                    mb_inds = self.batch_idx[start:end]

                    ratio, entropy, newvalue =\
                        self.get_mb_action_and_value(mb_inds)

                    pg_loss = self.compute_policy_loss(mb_inds,
                                                       ratio)

                    v_loss = self.compute_value_loss(mb_inds,
                                                     newvalue.view(-1))

                    entropy_loss = entropy.mean()
                    
                    self.mb_entropy_loss[self.mb_number] =\
                        entropy_loss.item()

                    loss =\
                        pg_loss -\
                        self.ent_coef * entropy_loss +\
                        v_loss * self.vf_coef

                    self.mb_loss[self.mb_number] = loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(),
                                            self.max_grad_norm)
                    self.optimizer.step()        

                    self.mb_number += 1
            
            self.compute_avg_training_epoch_metrics(iteration)

    def init_training_epoch(self):

        self.mb_number = 0
        self.mb_clipfracs = np.zeros(self.total_n_mb)
        self.mb_old_approx_kl = np.zeros(self.total_n_mb)
        self.mb_approx_kl = np.zeros(self.total_n_mb)
        self.mb_pg_loss = np.zeros(self.total_n_mb)
        self.mb_v_loss = np.zeros(self.total_n_mb)
        self.mb_entropy_loss = np.zeros(self.total_n_mb)
        self.mb_loss = np.zeros(self.total_n_mb)

    def compute_avg_training_epoch_metrics(self,
                                           iteration):

        avg_clipfracs = np.mean(self.mb_clipfracs)
        avg_old_approx_kl = np.mean(self.mb_old_approx_kl)
        avg_approx_kl = np.mean(self.mb_approx_kl)
        abg_pg_loss = np.mean(self.mb_pg_loss)
        avg_v_loss = np.mean(self.mb_v_loss)
        avg_entropy_loss = np.mean(self.mb_entropy_loss)
        avg_loss = np.mean(self.mb_loss)

        self.vec_env.add_tb_scalar("charts/avg_clipfracs",
                                    avg_clipfracs)
        
        self.vec_env.add_tb_scalar("charts/avg_old_approx_kl",
                                    avg_old_approx_kl)
        
        self.vec_env.add_tb_scalar("charts/avg_approx_kl",
                                    avg_approx_kl)
        
        self.vec_env.add_tb_scalar("charts/abg_pg_loss",
                                    abg_pg_loss)
        
        self.vec_env.add_tb_scalar("charts/avg_v_loss",
                                    avg_v_loss)
        
        self.vec_env.add_tb_scalar("charts/avg_entropy_loss",
                                    avg_entropy_loss)

        self.vec_env.add_tb_scalar("charts/avg_loss",
                                    avg_loss)
        
        status_str =\
            f"\niteration: {iteration} " +\
            f"avg_clipfracs: {avg_clipfracs:.4f} " +\
            f"avg_old_approx_kl: {avg_old_approx_kl:.6f} " +\
            f"avg_approx_kl: {avg_approx_kl:.6f}\n" +\
            f"abg_pg_loss: {abg_pg_loss:.4f} " +\
            f"avg_v_loss: {avg_v_loss:.4f} " +\
            f"avg_entropy_loss: {avg_entropy_loss:.4f} " +\
            f"avg_loss: {avg_loss:.4f}\n"

        print(status_str)  

    def get_mb_action_and_value(self,
                                mb_inds):

        mb_actions = self.b_actions.long()[mb_inds]

        _, newlogprob, entropy, newvalue =\
            self.agent.get_action_and_value(self.b_obs[mb_inds],
                                            mb_actions)

        logratio = newlogprob - self.b_logprobs[mb_inds]
        ratio = logratio.exp()

        """
        calculate approx_kl 
        http://joschu.net/blog/kl-approx.html
        """
        with torch.no_grad():

            self.mb_old_approx_kl[self.mb_number] =\
                (-logratio).mean().item()

            self.mb_approx_kl[self.mb_number] =\
                ((ratio - 1) - logratio).mean().item()

            self.mb_clipfracs[self.mb_number] +=\
                ((ratio - 1.0).abs() >\
                self.clip_coef).float().mean().item()

        return ratio, entropy, newvalue

    def compute_policy_loss(self,
                            mb_inds,
                            ratio):
        
        mb_advantages = self.b_advantages[mb_inds]

        if self.norm_adv:

            mb_advantages =\
                (mb_advantages - mb_advantages.mean()) /\
                (mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio

        pg_loss2 =\
            -mb_advantages *\
                torch.clamp(ratio,
                            1 - self.clip_coef,
                            1 + self.clip_coef)

        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        self.mb_pg_loss[self.mb_number] = pg_loss.item()

        return pg_loss
    
    def compute_value_loss(self,
                            mb_inds,
                            newvalue1d):

        if self.clip_vloss:

            v_loss_unclipped = (newvalue1d - self.b_returns[mb_inds]) ** 2

            v_clipped =\
                self.b_values[mb_inds] +\
                torch.clamp(newvalue1d- self.b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef)

            v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        #-------------------------------------------------------------
        else:
            v_loss =\
                0.5 * ((newvalue1d - self.b_returns[mb_inds]) ** 2).mean()

        self.mb_v_loss[self.mb_number] = v_loss.item()

        return v_loss