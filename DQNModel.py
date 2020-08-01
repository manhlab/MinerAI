# -*- coding: utf-8 -*-
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import numpy as np

# from keras.models import Sequential
# from keras.models import model_from_json
# from keras.layers import Dense, Activation
# from keras import optimizers
# from keras import backend as K
# import tensorflow as tf
from random import random, randrange
import os.path

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ReplayBuffer
from dqn_network import DQNetwork
from random import random, randrange


# Deep Q Network off-policy
class DQN:
    def __init__(
        self,
        input_dims=198,
        n_actions=6,
        gamma=0.1,
        epsilon=0.9,
        lr=0.0005,
        mem_size=10000,
        batch_size=32,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        algo="dnqagent",
        env_name="minerai",
        chkpt_dir="tmp/dqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0
        self.n_actions = n_actions
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name=self.env_name + "_" + self.algo + "_q_eval",
            chkpt_dir=self.chkpt_dir,
        )
        if os.path.exists(self.q_eval.checkpoint_file):
            self.q_eval.load_checkpoint()

        self.q_next = DQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name=self.env_name + "_" + self.algo + "_q_next",
            chkpt_dir=self.chkpt_dir,
        )
        if os.path.exists(self.q_next.checkpoint_file):
            self.q_next.load_checkpoint()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(
                self.q_eval.device
            )
            actions = self.q_eval.forward(state, self.get_state2(observation))
            action = torch.argmax(actions).item()
        else:
            action = randrange(self.n_actions)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states, self.get_state2(states))[indices, actions]
        q_next = self.q_next.forward(states_, self.get_state2(states_)).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def get_state2(self, observation):
        observation = np.array(observation).reshape(-1, 198)
        for i in range(observation.shape[0]):
            observation[
                i, min(int(observation[i, 192]) + int(observation[i, 193]) * 9, 192)
            ] = 1000
            observation[
                i, min(int(observation[i, 194]) + int(observation[i, 195]) * 9, 192)
            ] = 1000
            observation[
                i, min(int(observation[i, 196]) + int(observation[i, 197]) * 9, 192)
            ] = 1000
            observation[
                i, min(int(observation[i, 189]) + int(observation[i, 190]) * 9, 192)
            ] = 10000

        return (
            torch.tensor([observation], dtype=torch.float)
            .to(self.q_eval.device)
            .view(-1, 198)
        )
