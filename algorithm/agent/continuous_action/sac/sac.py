import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.agent import Agent
from utils import wandb_log
import utils

import hydra


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        wandb_log('train_critic/loss', critic_loss.item(), step)                                         

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(step)

       

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        wandb_log('train_actor/loss', actor_loss.item(), step)
        wandb_log('train_actor/target_entropy', self.target_entropy, step)
        wandb_log('train_actor/entropy', -log_prob.mean().item(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            wandb_log('train_alpha/loss', alpha_loss.item(), step)
            wandb_log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        g, s, a, r, s_, success = replay_buffer.get_batch(
            self.batch_size)
        obs = np.concatenate([s, g], axis=1)
        obs_ = np.concatenate([s_, g], axis=1)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_ = torch.as_tensor(obs_, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        not_done = 1 - success
        not_done =torch.as_tensor(not_done, dtype=torch.float32, device=self.device)
        # print("shapes: ",obs.shape,obs_.shape, a.shape, r.shape, success.shape )
        wandb_log('train/batch_reward', r.mean().item(), step)
        
        self.update_critic(obs, a, r, obs_,not_done , step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,self.critic_tau)

    def state_dict(self):
        """Return state dict for checkpointing (actor, critic, target, log_alpha, optimizers)."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        """Load from a checkpoint state dict."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.log_alpha.data = state["log_alpha"].to(self.device)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha_optimizer.load_state_dict(state["log_alpha_optimizer"])