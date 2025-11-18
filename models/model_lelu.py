# model.py
import torch
import torch.nn as nn

class LELU(nn.Module):
    def __init__(self, init_beta=0.5):
        super().__init__()
        # Learnable beta parameter
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float))

    def forward(self, x):
        # x > 0: identity
        pos = torch.relu(x)
        # x <= 0: custom exponential branch
        neg = torch.where(
            x <= 0,
            x * torch.exp((1 - self.beta) * x) - 1 + self.beta * x,
            torch.zeros_like(x)
        )
        return pos + neg

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            LELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)