import numpy as np
import matplotlib.pyplot as plt

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
plt.figure(figsize=(5, 4))
plt.plot(curve_points_w2[:,0], curve_points_w2[:,1], label='w = 0.3', color='k', linestyle = '-.')
plt.plot(curve_points_w1[:,0], curve_points_w1[:,1], label='w = 1.0', color='k')
plt.plot(curve_points_w0[:,0], curve_points_w0[:,1], label='w = 5.0', color='k', linestyle = ':')

plt.plot(*zip(*ctrlpts), label='Control points', color='red', linestyle = '--', marker ='o')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.ylim(-0.25, 11)
plt.grid(True, linewidth=0.1)
plt.savefig('PICS/nurbs.pdf')