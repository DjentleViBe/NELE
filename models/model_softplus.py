# model.py
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)