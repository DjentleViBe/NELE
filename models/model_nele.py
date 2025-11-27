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
        self.l = nn.Parameter(torch.tensor(-1.0))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.y1 = nn.Parameter(torch.tensor(-0.1))
        self.x1 = nn.Parameter(torch.tensor(-0.1))
        self.y0 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mask = x > 0
        device = x.device
        t = torch.linspace(0, 1, 200).to(device)
        N0 = (1 - t)**3
        N1 = 3 * t * (1 - t)**2
        N2 = 3 * t**2 * (1 - t)
        N3 = t**3

        cp0 = torch.tensor([1.0, -0.1], device=device)
        cp1 = torch.tensor([-0.5, -0.1], device=device)
        cp2 = torch.tensor([-1.0, -1.0], device=device)
        cp3 = torch.tensor([0.0, 0.0], device=device)
        cp0[0] = x.min()
        cp0[1] = self.y0
        cp1[0] = self.x1
        cp1[1] = self.y1
        cp2[0] = self.l / 1.4142
        cp2[1] = self.l / 1.4142
        
        control_points = torch.stack([cp0, cp1, cp2, cp3])  # shape (3, 2)

        # Weights
        weights = torch.ones(4, device=self.w1.device)
        weights[1] = self.w1   
        weights[2] = self.w2  
        # Numerator (weighted sum of control points)
        numerator = (N0[:, None] * weights[0] * control_points[0] +
                 N1[:, None] * weights[1] * control_points[1] +
                 N2[:, None] * weights[2] * control_points[2] + 
                 N3[:, None] * weights[3] * control_points[3])
        denominator = (N0 * weights[0] + N1 * weights[1] + N2 * weights[2]  + N3 * weights[3])[:, None]
    
        curve_points = numerator / (denominator + 1e-6)
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
        return torch.where(mask, x, y_queries)
    
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