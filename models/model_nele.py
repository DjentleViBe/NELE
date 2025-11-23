import torch
import torch.nn as nn

class NELE(nn.Module):
    def __init__(self, num_features=1, num_points=3, degree=2):
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
        B, C, H, W = x.shape
        device = x.device

        # Flatten spatial dims: (N, C)
        x_flat = x.permute(0,2,3,1).reshape(-1, C)  # (N, C)
        N = x_flat.shape[0]

        # Normalize input to [0,1] per channel
        x_min, x_max = x_flat.min(dim=0, keepdim=True)[0], x_flat.max(dim=0, keepdim=True)[0]
        u = (x_flat - x_min) / (x_max - x_min + 1e-6)  # (N, C)

        # Quadratic B-spline knots
        degree = 2
        n_knots = self.num_points + degree + 1
        knots = torch.linspace(-3, 3, n_knots, device=device)  # (num_points+3)

        # Compute all degree-0 basis functions: shape (N, C, num_points)
        B0 = ((u.unsqueeze(2) >= knots[:self.num_points].view(1,1,-1)) & 
              (u.unsqueeze(2) < knots[2:self.num_points+2].view(1,1,-1))).float()

        # For quadratic (degree-2), approximate using weighted sum of neighbors
        # Shifted versions of B0
        B0_pad = torch.cat([B0[:,:,:1], B0, B0[:,:,-1:]], dim=2)  # pad edges
        # Approx degree-2 basis using simple convolution along control points
        B0_pad = torch.cat([B0[:,:,:1], B0, B0[:,:,-1:]], dim=2)  # pad edges
        B2 = 0.25*B0_pad[:, :, :-2] + 0.5*B0_pad[:, :, 1:-1] + 0.25*B0_pad[:, :, 2:]
        
        # Apply learnable weights and control points
        cp = self.control_points.unsqueeze(0)  # (1, C, num_points)
        w  = self.weights.unsqueeze(0)         # (1, C, num_points)

        numerator = B2 * w * cp
        denominator = B2 * w + 1e-6

        y_flat = numerator.sum(dim=2) / denominator.sum(dim=2)  # (N, C)

        # Reshape back to (B, C, H, W)
        y = y_flat.view(B,H,W,C).permute(0,3,1,2).contiguous()
        return y

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