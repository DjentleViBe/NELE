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
         # learnable input/output scaling
        self.in_shift  = nn.Parameter(torch.zeros(1))
        self.in_scale  = nn.Parameter(torch.ones(1))
        self.out_shift = nn.Parameter(torch.zeros(1))
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_norm = (x - self.in_shift) / (self.in_scale.abs() + 1e-6)
        x_norm = torch.clamp(x_norm, 0, 1)
        one_minus_t = 1 - x_norm

        N0 = one_minus_t * one_minus_t
        N1 = 2 * x_norm * one_minus_t
        N2 = x_norm * x_norm

        # numerator and denominator directly
        numerator = N0 * self.weights[:, 0] * self.control_points[:, 0] + \
                    N1 * self.weights[:, 1] * self.control_points[:, 1] + \
                    N2 * self.weights[:, 2] * self.control_points[:, 2]

        denominator = N0 * self.weights[:, 0] + \
                    N1 * self.weights[:, 1] + \
                    N2 * self.weights[:, 2]
        y_norm = numerator / (denominator + 1e-6)
        return y_norm * self.out_scale + self.out_shift

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