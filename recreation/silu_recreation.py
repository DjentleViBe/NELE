import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------
# NURBS generator (cubic, 4 ctrl pts)
# ---------------------------
def nurbs_gen(control_points, weights, t):
    cp = control_points.reshape(4, 2)
    w = weights

    N0 = (1 - t)**3
    N1 = 3 * t * (1 - t)**2
    N2 = 3 * t**2 * (1 - t)
    N3 = t**3

    numerator = (N0[:, None] * w[0] * cp[0] +
                 N1[:, None] * w[1] * cp[1] +
                 N2[:, None] * w[2] * cp[2] +
                 N3[:, None] * w[3] * cp[3])

    denominator = (N0 * w[0] +
                   N1 * w[1] +
                   N2 * w[2] +
                   N3 * w[3])[:, None]

    return numerator / (denominator + 1e-12)

# ---------------------------
# silu
# ---------------------------
def silu(x):
    return x / (1 + np.exp(-x))

# target segment
x_target = np.linspace(-4, 0, 200)
y_target = silu(x_target)

x_extra = np.linspace(0, 4, 200)
y_extra = silu(x_extra)

# reparameterize into t ∈ [0, 1]
t = (x_target - x_target.min()) / (x_target.max() - x_target.min())

# ---------------------------
# Optimization objective
# ---------------------------
def loss(params):
    # params = 4 ctrl points * 2 coords + 4 weights = 12 values
    ctrl = params[:8].reshape(4, 2)
    w = params[8:]

    curve = nurbs_gen(ctrl, w, t)

    # match x and y separately
    x_cur, y_cur = curve[:, 0], curve[:, 1]

    # target x is known: linear from -4 to 0
    x_desired = x_target

    return np.mean((x_cur - x_desired)**2 + (y_cur - y_target)**2)

# ---------------------------
# Initial guess
# ---------------------------
initial_ctrl = np.array([
    [-4, 0],
    [-2.2, -0.02],
    [-0.7, -0.08],
    [0, 0]
]).flatten()

initial_weights = np.array([1.0, 0.9, 0.7, 1.0])

init_params = np.concatenate([initial_ctrl, initial_weights])

# ---------------------------
# Run optimization
# ---------------------------
res = minimize(
    loss,
    init_params,
    method='L-BFGS-B',
    options={'maxiter': 500}
)

opt_params = res.x
opt_ctrl = opt_params[:8].reshape(4, 2)
opt_w = opt_params[8:]

np.set_printoptions(precision=4, suppress=True)
print("Optimized Control Points:\n", opt_ctrl)
print("Optimized Weights:\n", opt_w)

opt_ctrl_mnist = np.array([
    [-1, 0],
    [-0.1, -0.1],
    [-1/2**0.5, -1/2**0.5],
    [0, 0]
])
opt_w_mnist = np.array([1.0, 1.0, 1.0, 1.0])
linear = np.array([[0, 0], [4, 4]])
linear_2 = np.array([[-1, 0], [-4, 0]])
# ---------------------------
# Plot results
# ---------------------------
curve = nurbs_gen(opt_ctrl, opt_w, t)
curve_mnist = nurbs_gen(opt_ctrl_mnist, opt_w_mnist, t)
plt.figure(figsize=(5, 4))
plt.plot(np.concatenate([x_target, x_extra]), np.concatenate([y_target, y_extra]), '--', label="silu", color = 'red')
plt.plot(curve[:, 0], curve[:, 1], label = 'silu approximation', color = 'k', linewidth=0.7)
#plt.plot(curve_mnist[:, 0], curve_mnist[:, 1], label="NURBS MNIST", color = 'green', linewidth=1.0)
plt.scatter(opt_ctrl[:, 0], opt_ctrl[:, 1], label="Control Points NURBS", color = 'k', s=3)
#plt.scatter(opt_ctrl_mnist[:, 0], opt_ctrl_mnist[:, 1], label="Control Points MNIST", color = 'green', marker = '+')
#plt.plot(linear[:, 0], linear[:, 1], color = 'green', linewidth=1.0)
#plt.plot(linear_2[:, 0], linear_2[:, 1], color = 'green', linewidth=1.0)

plt.legend()
plt.grid(True, linewidth = 0.2)
#plt.title("Optimized Cubic NURBS Approximation of Gsilu")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.axis('equal')
plt.savefig('./PICS/Approx_silu.pdf')