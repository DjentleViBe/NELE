import torch 
import torch.nn as nn
import math
import numpy as np
from scipy.optimize import bisect

class BELU(nn.Module):
    def __init__(self, num_points=4):
        super().__init__()
        # Each control point has (x, y)
        self.control_points = nn.Parameter(torch.zeros(num_points, 2))

    def bernstein_poly(self, i, n, t):
        comb = math.comb(n, i)
        return comb * (t ** i) * ((1 - t) ** (n - i))

    def forward(self, x):
        """
        x: tensor of shape (batch_size, 1)
        Returns y(t) for each input using Bezier control points
        """
        # Normalize x into [0, 1] as t
        t = (x + 1) / 2.0  # shape: (batch_size, 1)
        n = self.control_points.size(0) - 1

        # Compute Bezier y(t) for the whole batch
        y = torch.zeros_like(t)
        for i in range(n + 1):
            b = self.bernstein_poly(i, n, t)  # shape: (batch_size, 1)
            y += b * self.control_points[i, 1]  # only use y-coordinates
        return y
    
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            BELU(6),
            nn.Linear(64, 64),
            BELU(6),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)