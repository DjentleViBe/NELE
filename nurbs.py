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
    knots = np.linspace(0, 1, n + degree + 2)  # uniform knot vector
    numerator = np.zeros(2)
    denominator = 0.0
    for i in range(n+1):
        Ni = N(i, degree, u, knots)
        numerator += Ni * weights[i] * np.array(ctrlpts[i])
        denominator += Ni * weights[i]
    return numerator / denominator

# Tangent approximation using finite differences
def nurbs_tangent(ctrlpts, weights, degree, u, delta=1e-5):
    p1 = nurbs_curve(ctrlpts, weights, degree, u)
    p2 = nurbs_curve(ctrlpts, weights, degree, u+delta)
    return (p2 - p1) / delta

# Control points and weights
ctrlpts = [
    [-10, 0],
    [0, 0],
    [10, 1],  # notch point
]
weights = [1.0, 1.0, 1.0]
degree = 3

# Evaluate curve points
u_vals = np.linspace(0, 1, 200)
curve_points = np.array([nurbs_curve(ctrlpts, weights, degree, u) for u in u_vals])

# Tangent at notch
u_notch = 0.5
tangent = nurbs_tangent(ctrlpts, weights, degree, u_notch)
point = nurbs_curve(ctrlpts, weights, degree, u_notch)

# Plotting
plt.figure(figsize=(5, 5))
plt.plot(curve_points[:,0], curve_points[:,1], 'b-', label='NURBS curve', color='k')
plt.plot(*zip(*ctrlpts), 'ro--', label='Control points', color='red')
#plt.quiver(point[0], point[1], tangent[0], tangent[1],
#           color='g', scale=5, width=0.01, label='Tangent at notch')
#plt.scatter(point[0], point[1], color='k', zorder=5, label='Notch point')
plt.title("NURBS Curve from Scratch with Tangent")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.ylim(-1, 2)
plt.grid(True)
plt.show()