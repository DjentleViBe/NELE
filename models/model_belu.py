import torch 
import torch.nn as nn
import math

class BELU(nn.Module):
    def __init__(self, num_points=4):
        super().__init__()
        # Each control point has (x, y)
        # Shape: (num_points, 2)
        self.control_points = nn.Parameter(torch.zeros(num_points, 2))

    def bernstein_poly(self, n, t):
        """
        Compute all Bernstein basis polynomials B_i^n(t) for i=0..n
        t: tensor of shape (batch_size, num_features)
        Returns: tensor of shape (batch_size, num_features, n+1)
        """
        i = torch.arange(n + 1, device=t.device).view(1, 1, -1)  # shape (1,1,n+1)
        # t shape: (batch_size, num_features, 1)
        t = t.unsqueeze(-1)
        comb = torch.tensor([math.comb(n, j) for j in range(n+1)], dtype=t.dtype, device=t.device)
        comb = comb.view(1, 1, -1)
        B = comb * (t ** i) * ((1 - t) ** (n - i))
        return B  # shape: (batch_size, num_features, n+1)

    def forward(self, x):
        """
        x: shape (batch_size, num_features)
        Returns: shape (batch_size, num_features)
        """
        t = (x + 5) / 10.0  # normalize x into [0,1]
        batch_size, num_features = x.shape
        n = self.control_points.size(0) - 1

        # Compute Bernstein basis
        B = self.bernstein_poly(n, t)  # shape: (batch_size, num_features, n+1)

        # Control points y-coordinates
        y_cp = self.control_points[:, 1].view(1, 1, -1)  # shape (1,1,n+1)

        # Compute Bezier output via batch matrix multiplication
        y = (B * y_cp).sum(dim=-1)  # shape: (batch_size, num_features)
        return y
    
class Net(nn.Module):
    def __init__(self, input_dim, bezier_points):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            BELU(bezier_points),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)