# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
"""
NELE Activation Function (AF)
"""
import torch
from torch import nn
import config as cfg

class NELE(nn.Module):
    """
    NELE AF with learnable parameters and xmin derived
    """
    def __init__(self):
        super().__init__()

        # Per-feature control points and weights
        # Shape: (num_features, num_points)
        self.l = nn.Parameter(torch.tensor(-1.0))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.y1 = nn.Parameter(torch.tensor(-0.1))
        self.x1 = nn.Parameter(torch.tensor(-0.1))
        self.y0 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Forward function
        """
        mask = x > 0
        device = x.device
        t = torch.linspace(0, 1, 200).to(device)
        n0_val = (1 - t)**3
        n1_val = 3 * t * (1 - t)**2
        n2_val = 3 * t**2 * (1 - t)
        n3_val = t**3

        cp0 = torch.tensor([1.0, -0.1], device=device)
        cp0[0] = x.min()
        cp0[1] = self.y0
        cp1 = torch.tensor([-0.5, -0.1], device=device)
        cp1[0] = self.x1
        cp1[1] = self.y1
        cp2 = torch.tensor([-1.0, -1.0], device=device)
        cp2[0] = self.l / 1.4142
        cp2[1] = self.l / 1.4142
        cp3 = torch.tensor([0.0, 0.0], device=device)
        control_points = torch.stack([cp0, cp1, cp2, cp3])  # shape (3, 2)

        # Weights
        weights = torch.ones(4, device=self.w1.device)
        weights[1] = self.w1
        weights[2] = self.w2
        # Numerator (weighted sum of control points)
        numerator = (n0_val[:, None] * weights[0] * control_points[0] +
                 n1_val[:, None] * weights[1] * control_points[1] +
                 n2_val[:, None] * weights[2] * control_points[2] +
                 n3_val[:, None] * weights[3] * control_points[3])
        denominator = (n0_val * weights[0] + n1_val * weights[1] + \
                        n2_val * weights[2]  + n3_val * weights[3])[:, None]

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

class NeleLut(nn.Module):
    """
    NELE AF with buffer, learnable parameters and masking positive, xmin clamped
    """
    def __init__(self, num_points=16, x_min=-1.0, x_max=0.0):
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
        """
        Forward function
        """
        mask = x <= 0
        x_neg = x[mask]

        # Control points
        cp0_x, cp0_y = self.x_min, self.y0
        cp1_x, cp1_y = self.x1, self.y1
        cp2_val = self.l / 1.4142
        cp2_x, cp2_y = cp2_val, cp2_val

        # Weights
        w0, w1, w2, w3 = 1.0, self.w1, self.w2, 1.0
        # Compute Bézier curve
        n0_val, n1_val, n2_val, n3_val = self.N0, self.N1, self.N2, self.N3
        numerator_y = (
            n0_val*w0*cp0_y +
            n1_val*w1*cp1_y +
            n2_val*w2*cp2_y
        )
        numerator_x = (
            n0_val*w0*cp0_x +
            n1_val*w1*cp1_x +
            n2_val*w2*cp2_x
        )
        # Denominator: scalar sum
        denominator = n0_val * w0 + n1_val * w1 + n2_val * w2 + n3_val * w3 + 1e-12

        # LUT y values
        x_lut = numerator_x / denominator
        y_lut = numerator_y / denominator
        # Linear interpolation
        # Vectorized linear interpolation
        # print(x_lut.shape)
        x_min_val = x_lut[0]  # scalar tensor
        x_max_val = x_lut[-1]  # scalar tensor

        scale = (self.num_points - 1) / (x_max_val - x_min_val)
        indices = ((x_neg - x_min_val) * scale).clamp(0, self.num_points - 2)
        idx_lower = indices.floor().long()
        idx_upper = idx_lower + 1
        alpha = indices - idx_lower.float()

        y_lower = y_lut[idx_lower]
        y_upper = y_lut[idx_upper]
        y_neg = y_lower + alpha * (y_upper - y_lower)
        # Scatter back
        y_out = x.clone()
        y_out[mask] = y_neg

        return y_out

class NeleLutParam(nn.Module):
    """
    NELE AF with fixed params, buffers, double masking, xmin derived
    """
    def __init__(self, device, num_points = 200):
        super().__init__()
        self.cp1 = torch.tensor(cfg.cp1, device=device)
        self.cp2 = torch.tensor(cfg.cp2, device=device)
        self.cp3 = torch.tensor(cfg.cp3, device=device)
        self.w0 = torch.tensor(cfg.w0, device=device)
        self.w1 = torch.tensor(cfg.w1, device=device)
        self.w2 = torch.tensor(cfg.w2, device=device)
        self.w3 = torch.tensor(cfg.w3, device=device)
        t = torch.linspace(0, 1, num_points)
        self.register_buffer('t', t)
        self.register_buffer('N0', (1 - t)**3)
        self.register_buffer('N1', 3 * t * (1 - t)**2)
        self.register_buffer('N2', 3 * t**2 * (1 - t))
        self.register_buffer('N3', t**3)

    def forward(self, x):
        """
        Forward function
        """
        device = x.device
        cp0 = torch.tensor(cfg.cp0, device=device)
        cp0[0] = x.min()
        # Numerator (weighted sum of control points)
        numerator = (self.N0[:, None] * self.w0 * cp0 +
                 self.N1[:, None] * self.w1 * self.cp1 +
                 self.N2[:, None] * self.w2 * self.cp2 +
                 self.N3[:, None] * self.w3 * self.cp3)
        denominator = (self.N0 * self.w0 + self.N1 * self.w1 + \
                       self.N2 * self.w2  + self.N3 * self.w3)[:, None]

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

        slope = (y1 - y0) / (x1 - x0 + 1e-6)
        y_queries = y0 + slope * (t_queries_clamped - x0)
        return torch.where(x > 0, x,
                       torch.where(x < cp0[0], torch.zeros_like(x), y_queries))

class NeleLutParamDir(nn.Module):
    """
    NELE AF with fixed params, double masking, buffers and xmin clamped
    """
    def __init__(self, device, num_points = 200):
        super().__init__()
        self.num_points = num_points
        self.cp1 = torch.tensor(cfg.cp1, device=device)
        self.cp2 = torch.tensor(cfg.cp2, device=device)
        self.cp3 = torch.tensor(cfg.cp3, device=device)
        self.w0 = torch.tensor(cfg.w0, device=device)
        self.w1 = torch.tensor(cfg.w1, device=device)
        self.w2 = torch.tensor(cfg.w2, device=device)
        self.w3 = torch.tensor(cfg.w3, device=device)
        t = torch.linspace(0, 1, num_points)
        self.register_buffer('t', t)
        self.register_buffer('N0', (1 - t)**3)
        self.register_buffer('N1', 3 * t * (1 - t)**2)
        self.register_buffer('N2', 3 * t**2 * (1 - t))
        self.register_buffer('N3', t**3)

    def forward(self, x):
        """
        Forward function
        """
        device = x.device
        cp0 = torch.tensor(cfg.cp0, device=device)
        # cp0[0] = x.min()
        # Numerator (weighted sum of control points)
        numerator = (self.N0[:, None] * self.w0 * cp0 +
                 self.N1[:, None] * self.w1 * self.cp1 +
                 self.N2[:, None] * self.w2 * self.cp2 +
                 self.N3[:, None] * self.w3 * self.cp3)
        denominator = (self.N0 * self.w0 + self.N1 * self.w1 + \
                       self.N2 * self.w2  + self.N3 * self.w3)[:, None]

        curve_points = numerator / (denominator + 1e-6)
        # Linear interpolation in PyTorch
        x_lut = curve_points[:, 0]
        y_lut = curve_points[:, 1]

        x_min_val = x_lut[0]  # scalar tensor
        x_max_val = x_lut[-1]  # scalar tensor

        scale = (self.num_points - 1) / (x_max_val - x_min_val)
        indices = ((x - x_min_val) * scale).clamp(0, self.num_points - 2)
        idx_lower = indices.floor().long()
        idx_upper = idx_lower + 1
        alpha = indices - idx_lower.float()

        y_lower = y_lut[idx_lower]
        y_upper = y_lut[idx_upper]
        y_neg = y_lower + alpha * (y_upper - y_lower)
        return torch.where(x > 0, x,
                       torch.where(x < cp0[0], torch.zeros_like(x), y_neg))

class NeleLutLearn(nn.Module):
    """
    NELE AF with single masking, buffers, xmin derived,
    """
    def __init__(self, device, num_points = 200):
        super().__init__()
        self.w0 = torch.tensor(cfg.w0, device=device)
        self.w3 = torch.tensor(cfg.w3, device=device)
        self.l = nn.Parameter(torch.tensor(-1.0))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.y1 = nn.Parameter(torch.tensor(-0.1))
        self.x1 = nn.Parameter(torch.tensor(-0.1))
        self.y0 = nn.Parameter(torch.tensor(0.0))
        t = torch.linspace(0, 1, num_points)
        self.register_buffer('t', t)
        self.register_buffer('N0', (1 - t)**3)
        self.register_buffer('N1', 3 * t * (1 - t)**2)
        self.register_buffer('N2', 3 * t**2 * (1 - t))
        self.register_buffer('N3', t**3)

    def forward(self, x):
        """
        Forward function
        """
        device = x.device
        cp0 = torch.tensor(cfg.cp0, device=device)
        cp1 = torch.tensor(cfg.cp1, device=device)
        cp2 = torch.tensor(cfg.cp2, device=device)
        cp3 = torch.tensor(cfg.cp3, device=device)

        cp0[0] = x.min()
        cp0[1] = self.y0
        cp1[0] = self.x1
        cp1[1] = self.y1
        cp2[0] = self.l / 1.4142
        cp2[1] = self.l / 1.4142
        cp3 = torch.tensor([0.0, 0.0], device=device)

        # Numerator (weighted sum of control points)
        numerator = (self.N0[:, None] * self.w0 * cp0 +
                 self.N1[:, None] * self.w1 * cp1 +
                 self.N2[:, None] * self.w2 * cp2 +
                 self.N3[:, None] * self.w3 * cp3)
        denominator = (self.N0 * self.w0 + self.N1 * self.w1 + \
                       self.N2 * self.w2  + self.N3 * self.w3)[:, None]

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

        slope = (y1 - y0) / (x1 - x0 + 1e-6)
        y_queries = y0 + slope * (t_queries_clamped - x0)
        return torch.where(x > 0, x,
                        y_queries)
