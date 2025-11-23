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

    def N(self, i, p, u, knots):
        """
        Vectorized B-spline basis evaluation.
        u: (batch_size, num_features)
        knots: 1D tensor
        Returns: (batch_size, num_features)
        """
        if p == 0:
            return ((u >= knots[i]) & (u < knots[i+1])).float()
        denom1 = knots[i+p] - knots[i]
        denom2 = knots[i+p+1] - knots[i+1]
        term1 = torch.zeros_like(u)
        term2 = torch.zeros_like(u)
        if denom1 != 0:
            term1 = (u - knots[i]) / denom1 * self.N(i, p-1, u, knots)
        if denom2 != 0:
            term2 = (knots[i+p+1] - u) / denom2 * self.N(i+1, p-1, u, knots)
        return term1 + term2

    def forward(self, x):
        """
        x: (batch_size, num_features, H, W)
        Returns: (batch_size, num_features, H, W)
        """
        batch_size, num_features, H, W = x.shape
        device = x.device

        # Flatten spatial dimensions to (batch_size*H*W, num_features)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, num_features)  # (B*H*W, C)

        # Uniform knot vector per feature
        n = self.num_points - 1
        knots = torch.linspace(x_flat.min(), x_flat.max(), n + self.degree + 2, device=device)

        y = torch.zeros_like(x_flat)
        denom = torch.zeros_like(x_flat)

        # Evaluate NURBS per control point
        for i in range(self.num_points):
            Ni = self.N(i, self.degree, x_flat, knots)  # (B*H*W, C)
            cp = self.control_points[:, i].unsqueeze(0) # (1, C)
            w = self.weights[:, i].unsqueeze(0)         # (1, C)
            y += Ni * w * cp
            denom += Ni * w

        y = y / (denom + 1e-6)
        # Reshape back to (batch_size, C, H, W)
        y = y.reshape(batch_size, H, W, num_features).permute(0, 3, 1, 2).contiguous()
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