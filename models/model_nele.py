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

    def basis_functions(self, u, degree, knots):
        """
        u: (num_samples,)
        knots: (num_knots,)
        Returns: (num_samples, num_basis) B-spline basis for all u
        """
        num_basis = len(knots) - degree - 1
        B = torch.zeros(u.shape[0], num_basis, device=u.device)

        # Degree 0
        for i in range(num_basis):
            B[:, i] = ((u >= knots[i]) & (u < knots[i+1])).float()

        # Higher degrees
        for p in range(1, degree+1):
            B_prev = B.clone()
            for i in range(num_basis):
                denom1 = knots[i+p] - knots[i]
                denom2 = knots[i+p+1] - knots[i+1]

                term1 = torch.zeros_like(u)
                term2 = torch.zeros_like(u)
                if denom1 > 0:
                    term1 = (u - knots[i]) / denom1 * B_prev[:, i]
                if denom2 > 0 and i+1 < num_basis:
                    term2 = (knots[i+p+1] - u) / denom2 * B_prev[:, i+1]
                B[:, i] = term1 + term2
        return B  # (num_samples, num_basis)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        device = x.device

        n = self.num_points - 1
        knots = torch.linspace(x_flat.min(), x_flat.max(), n + self.degree + 2, device=device)

        # Compute basis functions for all samples and features
        # Ni_all: (B*H*W, C, num_points)
        Ni_all = torch.stack([self.basis_functions(x_flat[:, c], self.degree, knots) for c in range(C)], dim=1)

        # Broadcast weights and control points
        cp = self.control_points.unsqueeze(0)  # (1, C, num_points)
        w = self.weights.unsqueeze(0)          # (1, C, num_points)

        numerator = Ni_all * w * cp
        denominator = Ni_all * w

        y_flat = numerator.sum(dim=2) / (denominator.sum(dim=2) + 1e-6)
        y = y_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
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