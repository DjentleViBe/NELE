import torch 
import torch.nn as nn
import math


class BELU(nn.Module):
    def __init__(self, num_points=4):
        super().__init__()
        # Control points y-coordinates (learnable)
        # Shape: (num_points,)
        self.control_points = nn.Parameter(torch.zeros(num_points))

    def forward(self, x):
        """
        x: shape (batch_size, num_features)
        Returns: shape (batch_size, num_features)
        """
        # Normalize input to [0,1] assuming input range [-5,5]
        t = (x + 5) / 10.0
        t = t.clamp(0.0, 1.0)  # ensure valid range

        # De Casteljau's algorithm
        y = self.control_points.unsqueeze(0).unsqueeze(0)  # (1,1,num_points)
        n = y.size(-1)
        for r in range(1, n):
            y = (1 - t).unsqueeze(-1) * y[:, :, :-1] + t.unsqueeze(-1) * y[:, :, 1:]
        # y now shape: (batch_size, num_features, 1)
        return y.squeeze(-1)
    
class Net(nn.Module):
    def __init__(self, input_dim, bezier_points):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            BELU(bezier_points),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)