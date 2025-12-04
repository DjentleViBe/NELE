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

class NELE_LUT(nn.Module):
    def __init__(self, num_points=12, x_min=-1.0, x_max=1.0):
        super().__init__()
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max

        # Learnable parameters of the curve
        self.l = nn.Parameter(torch.tensor(-1.0))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.y1 = nn.Parameter(torch.tensor(-0.1))
        self.x1 = nn.Parameter(torch.tensor(-0.1))
        self.y0 = nn.Parameter(torch.tensor(0.0))

        # Precompute t for Bézier basis
        t = torch.linspace(0, 1, num_points)
        self.register_buffer('t', t)
        self.register_buffer('N0', (1 - t)**3)
        self.register_buffer('N1', 3 * t * (1 - t)**2)
        self.register_buffer('N2', 3 * t**2 * (1 - t))
        self.register_buffer('N3', t**3)

    def forward(self, x):
        mask = x > 0
        numerator = (
            self.N0*-0.1 +
            self.N1*self.w1*self.y1.item() +
            self.N2*self.w2*self.l.item() / 1.4142 +
            self.N3*0.0
        )
        denominator = (self.N0 + self.N1*self.w1 + self.N2*self.w2.item() + self.N3)
        y_lut = numerator / (denominator + 1e-6)
        # Vectorized linear interpolation
        scale = (self.num_points - 1) / (self.x_max - self.x_min)
        indices = ((x - self.x_min) * scale).clamp(0, self.num_points - 2)
        idx_lower = indices.floor().long()
        idx_upper = idx_lower + 1
        alpha = indices - idx_lower.float()

        y_lower = y_lut[idx_lower]
        y_upper = y_lut[idx_upper]
        y_out = y_lower + alpha * (y_upper - y_lower)

        return torch.where(mask, x, y_out)