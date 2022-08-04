import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models import Actor, Critic
import numpy as np


class DDPGAgent(object):
    def __init__(self, tau, input_dims,num_actions, gamma=0.99, max_size=1000000, hidden1_dims=400,
                 hidden2_dims=300, batch_size=64, critic_lr=0.0003, actor_lr=0.0003):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr


        ### Actor Networks ###
        self.actor = Actor(input_dims, num_actions, hidden1_dims, hidden2_dims, name='Actor')
        self.target_actor = Actor(input_dims, num_actions, hidden1_dims, hidden2_dims, name='Target_Actor')
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.actor_lr)

        ### Critic Networks ###
        self.critic = Critic(input_dims, num_actions, hidden1_dims, hidden2_dims, name='Critic')
        self.target_critic = Critic(input_dims, num_actions, hidden1_dims, hidden2_dims, name='Target_Critic')
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.critic_lr)


        self.noise = OUActionNoise(mu=np.zeros(num_actions))
        self.memory = ReplayBuffer(max_size, input_dims, num_actions)
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return(mu_prime.cpu().detach().numpy())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return()
        state, action, reward, new_state, terminal = \
            self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        terminal = T.tensor(terminal).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        
        target_actions = self.target_actor.forward(new_state)
        target_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        ### target with for loop ###
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*target_critic_value[j]*terminal[j])
        
        ### target vectorized ###
        # target = reward + self.gamma*target_critic_value*terminal

        
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)


        ### Critic update ###
        self.critic.train()
        self.critic_optim.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_optim.step()

        ### Actor update ###
        self.critic.eval()
        self.actor.train()
        self.actor_optim.zero_grad()
        mu = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor_optim.step()

        ### Target update ###
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class ReplayBuffer(object):
    def __init__(self, max_size, input_dims, num_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims[0]))
        self.new_state_memory = np.zeros((self.mem_size, input_dims[0]))
        self.action_memory = np.zeros((self.mem_size, num_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal