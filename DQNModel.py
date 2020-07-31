# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

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
from utils import Model4Gold

# Deep Q Network off-policy
class DQN: 
   
    def __init__(
            self,
            input_dim, #The number of inputs for the DQN network
            action_space, #The number of actions for the DQN network
            learning_rate = 0.00025, #The learning rate for the DQN network
            tau = 0.125, #The factor for updating the DQN target network from the DQN network
            model = None, #The DQN model
            target_model = None, #The DQN target model 
            device='cpu',
            accumulation_steps = 1):
      self.input_dim = input_dim
      self.action_space = action_space
      self.accumulation_steps = accumulation_steps
      self.epsilon=0.1
      self.gamma = 0.1
      self.epsilon_min = 0.05
      self.epsilon_decay = 0.9
      self.device = device
      
      #Creating networks
      self.model        = Model4Gold(self.action_space) #Creating the DQN model
      self.target_model = Model4Gold(self.action_space) #Creating the DQN target model

      ## optimizzer deploy
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
      optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 
      self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                  factor=0.5,
                                                                  patience=1,
                                                                  verbose=False,
                                                                  threshold=0.0001,
                                                                  threshold_mode='abs',
                                                                  cooldown=0,
                                                                  min_lr=1e-8,
                                                                  eps=1e-08)
      self.load_model()
     
    
    def act(self,state):     
      a_max = np.argmax(self.model(torch.tensor(state, dtype=torch.float)).detach().numpy())     
      if (random() < self.epsilon):
        a_chosen = randrange(self.action_space)
      else:
        a_chosen = a_max      
      return a_chosen
    
    
    def replay(self,samples,batch_size):
      inputs = np.zeros((batch_size, self.input_dim))
      targets = np.zeros((batch_size, self.action_space))
      
      for i in range(0,batch_size):
        state = samples[0][i,:]
        action = samples[1][i]
        reward = samples[2][i]
        new_state = samples[3][i,:]
        done= samples[4][i]
        
        inputs[i,:] = state
        targets[i,:] = self.target_model(torch.tensor(state, dtype=torch.float)).detach().numpy()       
        if done:
          targets[i,action] = reward
        else:
          Q_future = np.max(self.target_model(torch.tensor(state, dtype=torch.float)).detach().numpy())
          targets[i,action] = Q_future 
      #Training
      loss = self.train_on_batch(inputs, targets, batch_size) 



    def train_on_batch(self, inputs,target, batch_size):
        self.model.train()
        self.model.to(self.device)
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
       
        losses = 0
        for b_idx, data in enumerate(inputs):
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()
            data  = torch.tensor(data, dtype=torch.float).to(self.device)
            output = self.model(data)
            yt = torch.tensor(target[b_idx,:], dtype=torch.float).view(-1)
            loss = torch.nn.BCELoss()(output, yt)
            with torch.set_grad_enabled(True):
                loss.backward()
                if (b_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step(metrics=loss)
                    if b_idx > 0:
                        self.optimizer.zero_grad()
                losses += loss
        return losses/batch_size

    def target_train(self): 
      self.target_model = self.model
    
    
    def update_epsilon(self):
      self.epsilon =  self.epsilon*self.epsilon_decay
      self.epsilon =  max(self.epsilon_min, self.epsilon)
    
    
    def save_model(self,path, model_name):
        # serialize model to JSON
        torch.save(self.model.state_dict(), path + model_name + ".pt")
        print("Saved model to disk")


    def load_model(self):
      if os.path.isfile('/v/MinerAI/TrainedModels/DQNmodel_20200731-1454_ep1000.pt'):
        checkpoint = torch.load('/v/MinerAI/TrainedModels/DQNmodel_20200731-1454_ep1000.pt')
        self.model.load_state_dict(checkpoint)
        print("Loaded model from checkpoint success!!")
      else:
        print('none checkpoint! Train from beginning')
 

