import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model4Gold(nn.Module):
  def __init__(self, action_space):
    super(Model4Gold, self).__init__()
    self.conv1 = nn.Conv2d(1,1,3,stride=1, padding=1)
    self.conv2 = nn.Conv2d(1,1,3,stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(1)
    self.bn2 = nn.BatchNorm2d(1)

    self.fc1 = nn.Linear(204,8)
    self.fc2 =nn.Linear(11, action_space)
  def forward(self, s):
    s1 = F.relu(self.bn1(self.conv1(s[:189].view(1,1,21,9))))
    s1 = F.relu(self.bn2(self.conv2(s1)))

    s1 = s.view(-1)
    s1 = torch.cat((s1, s[189+3:]), dim=0)
    s1 = F.dropout(F.relu(self.fc1(s1).view(-1)))
    s1 = torch.cat((s1, s[189:189+3]), dim=0)
    out = self.fc2(s1)

    return F.softmax(out, dim=0)


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
