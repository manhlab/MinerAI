import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
class DQNetwork(nn.Module):
    def __init__(self, lr, action_space, name, input_dims, chkpt_dir):
        super(DQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}.bin")


        self.conv1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(195, 8)
        self.fc2 = nn.Linear(11, action_space)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, s):
        s1 = F.relu(self.bn1(self.conv1(s[:,:189].view(-1, 1, 21, 9))))
        s1 = F.relu(self.bn2(self.conv2(s1)))

        s1 = s1.view(-1, 21*9)
        s1 = torch.cat((s1, s[:, 189+3:]), dim=1)
        s1 = F.dropout(F.relu(self.fc1(s1).view(-1)))

        s1 = torch.cat((s1.view(-1, 8), s[:, 189:189+3]), dim=1)
        out = self.fc2(s1)

        return F.softmax(out, dim=1)
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))



class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
# class Model4Gold(nn.Module):
#   def __init__(self, action_space):
#     super(Model4Gold, self).__init__()
#     self.conv1 = nn.Conv2d(1,1,3,stride=1, padding=1)
#     self.conv2 = nn.Conv2d(1,1,3,stride=1, padding=1)
#     self.bn1 = nn.BatchNorm2d(1)
#     self.bn2 = nn.BatchNorm2d(1)

#     self.fc1 = nn.Linear(206,8)
#     self.fc2 =nn.Linear(9, action_space)
#   def forward(self, s):
#     s1 = F.relu(self.bn1(self.conv1(s[:189].view(1,1,21,9))))
#     s1 = F.relu(self.bn2(self.conv2(s1)))

#     s1 = s.view(-1)
#     s1 = torch.cat((s1, s[189:189+8]), dim=0)
#     s1 = F.dropout(F.relu(self.fc1(s1).view(-1)))
#     s1 = torch.cat((s1, s[189+8:]), dim=0)
#     out = self.fc2(s1)

#     return F.softmax(out, dim=0)
