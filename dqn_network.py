import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch


class DQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}.bin")

        self.conv1 = nn.Conv2d(1, 32, 2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 2, stride=1)

      

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512 * 2, n_actions)
        self.fc3 = nn.Linear(n_actions + 1, n_actions)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)



    def forward(self, state, state2):
        conv1 = F.relu(self.conv1(state[:, :189].view(-1,1,9, 21)))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))


        
        conv1s = F.relu(self.conv1(state2[:, :189].view(-1,1, 9, 21)))
        conv2s = F.relu(self.conv2(conv1s))
        conv3s = F.relu(self.conv3(conv2s))
        conv_state2 = conv3.view(conv3s.size()[0], -1)
        flat1s = F.relu(self.fc1(conv_state2))

        actions = F.relu(self.fc2(torch.cat((flat1, flat1s), dim=1)))
        actions = self.fc3(torch.cat((actions, state2[:, 191].view(-1, 1)), dim=1))
        # actions = self.fc2(flat1)
        return actions

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
