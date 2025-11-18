import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------
# n-degree Bézier curve function
# -------------------------------

def bernstein_poly(i, n, t):
    """Compute Bernstein polynomial for index i, degree n."""
    return math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, num_points=1000):
    """Compute an n-degree Bézier curve from control points."""
    n = len(control_points) - 1
    t_values = np.linspace(0.0, 1.0, num_points)
    curve = np.zeros((num_points, len(control_points[0])))

    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t_values), control_points[i])
    return curve

# Example: define any number of control points
control_points = np.array([
    [0.0, 0.0],
    [0.3, 0.8],
    [0.6, -0.5],
    [1.0, 1.0],
    [1.3, 0.2]  # <- you can add or remove points freely
])

# Compute the curve
curve = bezier_curve(control_points)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(curve[:, 0], curve[:, 1], 'b-', label='Bézier curve')
plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control points')
plt.title(f'{len(control_points)-1}-degree Bézier curve')
plt.legend()
plt.axis('equal')
plt.show()