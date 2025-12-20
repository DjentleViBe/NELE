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

class NeleUniversal(nn.Module):
    """
    NELE AF with hyperparam learning, masking options, buffer and xmin clamping
    """
    def __init__(self, device, num_points = 200):
        super().__init__()
        self.num_points = num_points
        self.device = device
        t = torch.linspace(0, 1, num_points)
        self.register_buffer('t', t)
        self.register_buffer('N0', (1 - t)**3)
        self.register_buffer('N1', 3 * t * (1 - t)**2)
        self.register_buffer('N2', 3 * t**2 * (1 - t))
        self.register_buffer('N3', t**3)
        if cfg.learnable:
            self.l = nn.Parameter(torch.tensor(cfg.cp2[0] * 1.4142))
            self.w1 = nn.Parameter(torch.tensor(cfg.w1))
            self.w2 = nn.Parameter(torch.tensor(cfg.w2))
            self.y1 = nn.Parameter(torch.tensor(cfg.cp1[1]))
            self.x1 = nn.Parameter(torch.tensor(cfg.cp1[0]))
            self.y0 = nn.Parameter(torch.tensor(0.0))
            self.x0 = nn.Parameter(torch.tensor(cfg.cp0[0]))
        else:
            self.cp1 = torch.tensor(cfg.cp1, device=device)
            self.cp2 = torch.tensor(cfg.cp2, device=device)
            self.cp3 = torch.tensor(cfg.cp3, device=device)
            self.w0 = torch.tensor(cfg.w0, device=device)
            self.w1 = torch.tensor(cfg.w1, device=device)
            self.w2 = torch.tensor(cfg.w2, device=device)
            self.w3 = torch.tensor(cfg.w3, device=device)

    def forward(self, x):
        if cfg.learnable == False:
            cp0 = torch.tensor(cfg.cp0, device=self.device)
            if cfg.clamping == True:
                cp0[0] = x.min()
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
            y = y_lower + alpha * (y_upper - y_lower)
        else:
            mask = x <= 0
            x_neg = x[mask]
            cp0_x, cp0_y = self.x0, self.y0
            if cfg.clamping == True:
                cp0_x = x.min()
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
            y = x.clone()
            y[mask] = y_neg

        # return
        if cfg.masking == 0:
            return y
        elif cfg.masking == 1:
            return torch.where(x > 0, x, y)
        else:
            return torch.where(x > 0, x, torch.where(x < cfg.cp0[0], torch.zeros_like(x), y))
