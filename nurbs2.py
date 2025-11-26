import torch
import matplotlib.pyplot as plt

control_points = ([0.0, -5.0, 20.0])
weights = ([1.0, -0.5, 1.0])

def forward(x):
        # Create mask for x > 0

        # Case where x <= 0
        t = (x - x.min()) / (x.max() - x.min())
        #x_norm = torch.clamp(x_norm, -0.1, 0.0)
        one_minus_t = 1 - t

        N0 = one_minus_t * one_minus_t
        N1 = 2 * t * one_minus_t
        N2 = t * t

        numerator = N0 * weights[0] * control_points[0] + \
                    N1 * weights[1] * control_points[1] + \
                    N2 * weights[2] * control_points[2]

        denominator = N0 * weights[0] + \
                    N1 * weights[1] + \
                    N2 * weights[2]

        y_norm = numerator / (denominator + 1e-6)
        #y_norm = y_norm * out_scale + out_shift
       
        # Use torch.where to choose elementwise between x and y_norm
        return y_norm

x = torch.linspace(-20, 20, 100)
y = forward(x)
plt.plot(x, y)
plt.savefig('nurbs.png')
