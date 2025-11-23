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
        one_minus_t = 1 - x

        N0 = one_minus_t * one_minus_t
        N1 = 2 * x * one_minus_t
        N2 = x * x

        # numerator and denominator directly
        numerator = N0 * self.weights[:, 0] * self.control_points[:, 0] + \
                    N1 * self.weights[:, 1] * self.control_points[:, 1] + \
                    N2 * self.weights[:, 2] * self.control_points[:, 2]

        denominator = N0 * self.weights[:, 0] + \
                    N1 * self.weights[:, 1] + \
                    N2 * self.weights[:, 2]

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