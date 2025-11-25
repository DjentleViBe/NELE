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
        self.control_points1 = -0.1
        self.control_points3 = nn.Parameter(torch.tensor([1.0]))
        self.weights = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x_norm = (x - x.min()) / (x.max() - x.min())
        one_minus_t = 1 - x_norm

        N0 = one_minus_t * one_minus_t
        N1 = 2 * x_norm * one_minus_t
        N2 = x_norm * x_norm

        numerator = N0 * 1.0 * self.control_points1 + \
                    N1 * self.weights * 0.0 + \
                    N2 * 1.0 * self.control_points3

        denominator = N0 * 1.0 + \
                    N1 * self.weights + \
                    N2 * 1.0

        y_norm = numerator / (denominator + 1e-6)
        return y_norm

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