"""
Contains the neural network architecture for NLSD
"""
from torch import nn

class Net(nn.Module):
    """
    Neural Network for NLSD
    """
    def __init__(self, input_dim, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            activation,
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward function
        """
        return self.net(x)
