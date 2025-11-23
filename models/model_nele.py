import torch
import torch.nn as nn

class NELE(nn.Module):
    def __init__(self, num_features=64, num_points=3, degree=2):
        super().__init__()
        self.num_features = num_features
        self.num_points = num_points
        self.degree = degree

        # Per-feature control points and weights
        # Shape: (num_features, num_points)
        self.control_points = nn.Parameter(
            torch.linspace(-1, 1, num_points).repeat(num_features, 1)
        )
        self.weights = nn.Parameter(torch.ones(num_features, num_points))

    def forward(self, x):
        """
        Ultra-fast hardcoded quadratic B-spline with 3 control points.
        x: (batch_size, num_features)
        Returns: (batch_size, num_features)
        """
        # Quadratic basis functions for uniform knot vector [0,0,0,1,1,1]
        # Closed-form expressions for degree 2, 3 control points
        one_minus_t = 1 - x
        N0 = one_minus_t * one_minus_t
        N1 = (x + x) * one_minus_t  # 2*t*(1-t) = t + t - 2*t*t, but this is faster
        N2 = x * x
        
        # Extract control points and weights
        N = torch.stack([N0, N1, N2], dim=-1)  # (batch_size, num_features, 3)
        numerator = (N * self.weights.unsqueeze(0) * self.control_points.unsqueeze(0)).sum(dim=-1)
        denominator = (N * self.weights.unsqueeze(0)).sum(dim=-1)
        return numerator / (denominator + 1e-6)

class Net(nn.Module):
    def __init__(self, input_dim, nurbs_points, degree):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            NELE(1, nurbs_points, degree),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)