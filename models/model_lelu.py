"""
LELU Activation function
"""
import torch
from torch import nn

class LELU(nn.Module):
    """
    Leaky ELU activation function
    """
    def __init__(self, init_beta=0.5):
        super().__init__()
        # Learnable beta parameter
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float))

    def forward(self, x):
        """
        Forward function
        """
        # x > 0: identity
        pos = torch.relu(x)
        # x <= 0: custom exponential branch
        neg = torch.where(
            x <= 0,
            torch.exp((1 - self.beta) * x) - 1 + self.beta * x,
            torch.zeros_like(x)
        )
        return pos + neg
