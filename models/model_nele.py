import torch
import torch.nn as nn
import config as cfg

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
    def __init__(self, num_points=16, x_min=-1.0, x_max=0.0):
        super().__init__()
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max

        # Precompute t for Bézier basis
        t = torch.linspace(0, 1, num_points)
        self.register_buffer('t', t)
        self.register_buffer('N0', (1 - t)**3)
        self.register_buffer('N1', 3 * t * (1 - t)**2)
        self.register_buffer('N2', 3 * t**2 * (1 - t))
        self.register_buffer('N3', t**3)

    def forward(self, x):
        mask = x <= 0
        x_neg = x[mask]
        
        # Control points
        cp0_x, cp0_y = cfg.cp0[0], cfg.cp0[1]
        cp1_x, cp1_y = cfg.cp1[0], cfg.cp1[1]
        cp2_val = -1.0 / 1.4142
        cp2_x, cp2_y = cp2_val, cp2_val

        # Weights
        w0, w1, w2, w3 = cfg.w0, cfg.w1, cfg.w2, cfg.w3
        # Compute Bézier curve
        N0, N1, N2, N3 = self.N0, self.N1, self.N2, self.N3
        numerator_y = (
            N0*w0*cp0_y +
            N1*w1*cp1_y +
            N2*w2*cp2_y
        )
        numerator_x = (
            N0*w0*cp0_x +
            N1*w1*cp1_x +
            N2*w2*cp2_x
        )
        # Denominator: scalar sum
        denominator = N0 * w0 + N1 * w1 + N2 * w2 + N3 * w3 + 1e-6

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

        return torch.where(x > 0, x,
                       torch.where(x < cfg.cp0[0], torch.zeros_like(x), y_out))
    
class NELE_LUT_PARAM(nn.Module):
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
        device = x.device
        cp0 = torch.tensor(cfg.cp0, device=device)
        cp0[0] = x.min()
        # Numerator (weighted sum of control points)
        numerator = (self.N0[:, None] * self.w0 * cp0 +
                 self.N1[:, None] * self.w1 * self.cp1 +
                 self.N2[:, None] * self.w2 * self.cp2 + 
                 self.N3[:, None] * self.w3 * self.cp3)
        denominator = (self.N0 * self.w0 + self.N1 * self.w1 + self.N2 * self.w2  + self.N3 * self.w3)[:, None]
    
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
    
class NELE_LUT_LEARN(nn.Module):
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
        denominator = (self.N0 * self.w0 + self.N1 * self.w1 + self.N2 * self.w2  + self.N3 * self.w3)[:, None]
    
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