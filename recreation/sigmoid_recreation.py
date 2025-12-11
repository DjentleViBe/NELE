import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------
# NURBS generator (cubic, 4 ctrl pts)
# ---------------------------
def nurbs_gen_sigmoid(control_points, weights, t):
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
# GELU
# ---------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# target segment
x_target = np.linspace(-4, 0, 200)
y_target = sigmoid(x_target)

x_extra = np.linspace(0, 4, 200)
y_extra = sigmoid(x_extra)

# reparameterize into t ∈ [0, 1]
t = (x_target - x_target.min()) / (x_target.max() - x_target.min())

# ---------------------------
# Optimization objective
# ---------------------------
def loss_sigmoid(params):
    # params = 4 ctrl points * 2 coords + 4 weights = 12 values
    ctrl = params[:8].reshape(4, 2)
    w = params[8:]

    curve = nurbs_gen_sigmoid(ctrl, w, t)

    # match x and y separately
    x_cur, y_cur = curve[:, 0], curve[:, 1]

    # target x is known: linear from -4 to 0
    x_desired = x_target

    return np.mean((x_cur - x_desired)**2 + (y_cur - y_target)**2)
