import torch
from torch import nn
import sys
import os

sys.path.append("..")


PATH = "C:/Users/lth95/Project/snake/data/model/cityscapes/197.pth"

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
net = MLP()
print(net.state_dict())
net = torch.load(PATH)
# net.load_state_dict(torch.load(PATH))
print("*"*30)
print(net)