import torch
import torch.nn as nn
from scipy.interpolate import interp1d

class NELE(nn.Module):
    def __init__(self, num_features=64, num_points=3, degree=2):
        super().__init__()
        self.num_features = num_features
        self.num_points = num_points
        self.degree = degree

        # Per-feature control points and weights
        # Shape: (num_features, num_points)
        
        self.middle_w = nn.Parameter(torch.tensor(1.0))
        self.middle_x = nn.Parameter(torch.tensor(0.0))
        self.middle_y = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        t = torch.linspace(0, 1, 20)
        N0 = (1 - t)**2
        N1 = 2 * t * (1 - t)
        N2 = t**2
        
        cp1 = torch.stack([self.middle_x, self.middle_y])
        cp0 = torch.tensor([-1.0, 0.0])
        cp2 = torch.tensor([1.0, 1.0])
        cp0[0] = x.min()
        cp2[0] = x.max()
        cp2[1] = x.max()
        control_points = torch.stack([cp0, cp1, cp2])  # shape (3, 2)

        # Weights
        weights = torch.ones(3, device=self.middle_w.device)
        weights[1] = self.middle_w      
        # Numerator (weighted sum of control points)
        numerator = (N0[:, None] * weights[0] * control_points[0] +
                 N1[:, None] * weights[1] * control_points[1] +
                 N2[:, None] * weights[2] * control_points[2])
        denominator = (N0 * weights[0] + N1 * weights[1] + N2 * weights[2])[:, None]
    
        curve_points = numerator / (denominator + 1e-12)
        # Linear interpolation in PyTorch
        x_vals = curve_points[:, 0]
        y_vals = curve_points[:, 1]

        # Ensure t_queries within the x range
        t_queries_clamped = torch.clamp(x, x_vals.min(), x_vals.max())

        # Get indices for linear interpolation
        idx = torch.searchsorted(x_vals.contiguous(), t_queries_clamped)
        idx = torch.clamp(idx, 1, len(x_vals)-1)

        x0 = x_vals[idx-1]
        x1 = x_vals[idx]
        y0 = y_vals[idx-1]
        y1 = y_vals[idx]

        slope = (y1 - y0) / (x1 - x0 + 1e-12)
        y_queries = y0 + slope * (t_queries_clamped - x0)
        return y_queries
    
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