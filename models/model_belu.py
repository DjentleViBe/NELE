import torch 
import torch.nn as nn

class BELU(nn.Module):
    def __init__(self, num_features=1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x * torch.sigmoid(self.beta[0] * x)
    
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            BELU(1),
            nn.Linear(64, 64),
            BELU(1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)