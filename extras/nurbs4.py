import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def nurbs_gen(control_points, weights):
    """
    Cubic NURBS with 4 control points
    """
    control_points = np.array(control_points)  # shape (4, 2)
    t = np.linspace(0, 1, 20)
    
    # Cubic Bernstein basis
    N0 = (1 - t)**3
    N1 = 3 * t * (1 - t)**2
    N2 = 3 * t**2 * (1 - t)
    N3 = t**3
    
    # Numerator (weighted sum of control points)
    numerator = (N0[:, None] * weights[0] * control_points[0] +
                 N1[:, None] * weights[1] * control_points[1] +
                 N2[:, None] * weights[2] * control_points[2] +
                 N3[:, None] * weights[3] * control_points[3])
    
    # Denominator (weighted sum of basis functions)
    denominator = (N0 * weights[0] + N1 * weights[1] + N2 * weights[2] + N3 * weights[3])[:, None]
    
    curve_points = numerator / (denominator + 1e-12)
    return curve_points

def nurbs_point(control_points, weights, t):
    control_points = np.array(control_points)
    
    # Quadratic Bernstein basis
    N0 = (1 - t)**3
    N1 = 3 * t * (1 - t)**2
    N2 = 3 * t**2 * (1 - t)
    N3 = t**3
    
    # Numerator and denominator
    numerator = (N0 * weights[0] * control_points[0] +
                 N1 * weights[1] * control_points[1] +
                 N2 * weights[2] * control_points[2] +
                 N3 * weights[3] * control_points[3])
    
    denominator = (N0 * weights[0] + N1 * weights[1] + N2 * weights[2] + N3 * weights[3])
    
    point = numerator / (denominator + 1e-12)
    return point  # Returns [x, y]

# Control points and weights
l = 6
ctrlpts = [
    [-10, 0],
    [-5, -1],
    [-l/1.41, -l/1.41],
    [0, 0]
]
weights_0 = [1.0, 1.0, 1.0, 1.0]

# Evaluate curve points
curve_points_w0 = nurbs_gen(ctrlpts, weights_0)
t_val = 0.3
point = nurbs_point(ctrlpts, weights_0, t_val)

# Plot
plt.plot(curve_points_w0[:, 0], curve_points_w0[:, 1], label='NURBS Curve')
from scipy.interpolate import interp1d

# curve_points_w0 from your nurbs_gen function
x_vals = curve_points_w0[:, 0]
y_vals = curve_points_w0[:, 1]
interp_func = interp1d(x_vals, y_vals, kind='cubic')
x_queries = np.array([-5, -6, -3, -2])
y_queries = interp_func(x_queries)
plt.scatter(point[0], point[1], color='green', label=f'Curve point t={t_val}')
plt.scatter(x_queries, y_queries, color='black', label=f'Curve point t={t_val}', s = 3)
plt.scatter(*zip(*ctrlpts), color='red', label='Control Points')
plt.legend()
plt.axis('equal')
plt.show()