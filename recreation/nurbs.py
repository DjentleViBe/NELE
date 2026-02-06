import numpy as np
import matplotlib.pyplot as plt
from nurbs_gen import nurbs_gen
# Recursive B-spline basis function
def N(i, p, u, knots):
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i+1] else 0.0
    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]
    term1 = 0.0
    term2 = 0.0
    if denom1 != 0:
        term1 = (u - knots[i]) / denom1 * N(i, p-1, u, knots)
    if denom2 != 0:
        term2 = (knots[i+p+1] - u) / denom2 * N(i+1, p-1, u, knots)
    return term1 + term2

# Evaluate NURBS curve
def nurbs_curve(ctrlpts, weights, degree, u):
    n = len(ctrlpts) - 1
    knots = np.concatenate((
    np.zeros(degree),
    np.linspace(0, 1, n - degree + 2),
    np.ones(degree)
    ))
    numerator = np.zeros(2)
    denominator = 0.0
    for i in range(n+1):
        Ni = N(i, degree, u, knots)
        numerator += Ni * weights[i] * np.array(ctrlpts[i])
        denominator += Ni * weights[i]
    return numerator / (denominator)

# Tangent approximation using finite differences
def nurbs_tangent(ctrlpts, weights, degree, u, delta=1e-5):
    p1 = nurbs_curve(ctrlpts, weights, degree, u)
    p2 = nurbs_curve(ctrlpts, weights, degree, u+delta)
    return (p2 - p1) / delta

# Control points and weights
ctrlpts = [
    [-10, 0],
    [0, 0],
    [10, 10],  # notch point
]
weights_2 = [1.0, 0.3, 1.0]
weights_1 = [1.0, 1.0, 1.0]
weights_0 = [1.0, 5.0, 1.0]
degree = 2

# Evaluate curve points
u_vals = np.linspace(0, 1, 200)
curve_points_w2 = np.array([nurbs_curve(ctrlpts, weights_2, degree, u) for u in u_vals])
curve_points_w1 = np.array([nurbs_curve(ctrlpts, weights_1, degree, u) for u in u_vals])
curve_points_w0 = np.array([nurbs_curve(ctrlpts, weights_0, degree, u) for u in u_vals])
# Plotting
x_target = np.linspace(-4, 0, 200)
t_val = (x_target - x_target.min()) / (x_target.max() - x_target.min())
ctrlpts_cubic = np.array([
    [-10, 0],
    [-5, 6],
    [0, 0],
    [10, 10],  # notch point
])
weights_cubic = np.array([1.0, 1.0, 1.0, 1.0])
cubic_points = nurbs_gen(ctrlpts_cubic, weights_cubic, t_val)
weights_cubic2 = np.array([1.0, 0.3, 0.3, 1.0])
cubic_points2 = nurbs_gen(ctrlpts_cubic, weights_cubic2, t_val)

fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ax1.plot(curve_points_w2[:,0], curve_points_w2[:,1], label='w = 0.3', color='k', linestyle = '-.', linewidth = 1.0)
ax1.plot(curve_points_w1[:,0], curve_points_w1[:,1], label='w = 1.0', color='k', linewidth = 1.0)
ax1.plot(curve_points_w0[:,0], curve_points_w0[:,1], label='w = 5.0', color='k', linestyle = ':', linewidth = 1.0)
ax1.plot(*zip(*ctrlpts), label='Control points', color='red', linestyle = '--', marker ='o', linewidth = 1.0)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.set_ylim(-0.25, 11)
ax1.grid(True, linewidth=0.1)
ax1.set_title("(a)")

ax2.plot(cubic_points[:,0], cubic_points[:,1], label='w = 1.0, 1.0', color='k', linewidth = 1.0)
ax2.plot(cubic_points2[:,0], cubic_points2[:,1], label='w = 0.3, 0.3', linestyle = '--', color='k', linewidth = 1.0)
ax2.plot(*zip(*ctrlpts_cubic), label='Control points', color='red', linestyle = '--', marker ='o', linewidth = 1.0)
ax2.grid(True, linewidth=0.1)
ax2.set_title("(b)")
ax2.legend()
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
plt.tight_layout()
plt.savefig('PICS/nurbs.pdf')