# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating GELU with NELE
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from recreation.nurbs_gen import nurbs_gen
from recreation.gen_loss import loss_af
import config as cfg
# ---------------------------
# GELU
# ---------------------------
def gelu(x):
    """
    GELU function
    
    :param x: X values
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_special():
    """
    Generates the gelu approximation using NELE and the settings used for MNIST
    """
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
    x_target = np.linspace(-4, 0, 200)
    t_val = (x_target - x_target.min()) / (x_target.max() - x_target.min())
    y_target = gelu(x_target)
    x_extra = np.linspace(0, 4, 200)
    y_extra = gelu(x_extra)
    # ---------------------------
    # Run optimization
    # ---------------------------
    res = minimize(
        loss_af,
        init_params,
        args=(t_val, x_target, y_target),
        method='L-BFGS-B',
        options={'maxiter': 500}
    )

    opt_params = res.x
    opt_ctrl = opt_params[:8].reshape(4, 2)
    opt_w = opt_params[8:]

    np.set_printoptions(precision=4, suppress=True)
    print("Optimized Control Points:\n", opt_ctrl)
    print("Optimized Weights:\n", opt_w)
    linear = np.array([[0, 0], [4, 4]])
    linear_2 = np.array([[-8, 0], [-4, 0]])
    curves = []
    for cp0x, cp0y, cp1x, cp1y, length, w_0, w_1, w_2, w_3 in product(
        cfg.range_cp0_x,
        cfg.range_cp0_y,
        cfg.range_cp1_x,
        cfg.range_cp1_y,
        cfg.range_l,
        cfg.range_w0,
        cfg.range_w1,
        cfg.range_w2,
        cfg.range_w3,
    ):
        cp0 = [cp0x, cp0y]
        cp1 = [cp1x, cp1y]
        cp2 = [length/1.4142, length/1.41422]
        cp3 = [0.0, 0.0]
        opt_ctrl_range = np.array([cp0, cp1, cp2, cp3])
        opt_w_range = np.array([w_0, w_1, w_2, w_3])
        curves.append(nurbs_gen(opt_ctrl_range, opt_w_range, t_val))
    curves = np.stack(curves)
    mean_curve = curves.mean(axis=0)
    lower = curves.min(axis=0)
    upper = curves.max(axis=0)
    # ---------------------------
    # Plot results
    # ---------------------------
    curve_gen = nurbs_gen(opt_ctrl, opt_w, t_val)
    #curve_mnist = nurbs_gen(opt_ctrl_mnist, opt_w_mnist, t_val)
    plt.figure(figsize=(5, 4))
    plt.rcParams['text.usetex'] = True
    plt.plot(np.concatenate([x_target, x_extra]), np.concatenate([y_target, y_extra]), \
            '--', label=r'$\texttt{GELU}$', color = 'red')
    plt.plot(curve_gen[:, 0], curve_gen[:, 1], label = r'$\texttt{GELU}$ approximation', \
            color = 'k', linewidth=0.7)
    plt.plot(curves[-1, :, 0], mean_curve[:, 1], label=r'$\texttt{NELE}$ experiment mean', \
            color = 'blue', linewidth=0.7)
    plt.fill_between(
    curves[-1, :, 0],
    lower[:, 1],
    upper[:, 1],
    alpha=0.3,
    label="Hyper-parameter variation band"
    )
    plt.scatter(opt_ctrl[:, 0], opt_ctrl[:, 1], label=r'Control Points $\texttt{GELU}$ approx', \
                color = 'k', s=3)
    plt.plot(linear[:, 0], linear[:, 1], color = 'blue', linewidth=0.7)
    plt.plot(linear_2[:, 0], linear_2[:, 1], color = 'blue', linewidth=0.7)
    plt.legend()
    plt.grid(True, linewidth = 0.2)
    #plt.title("Optimized Cubic NURBS Approximation of GELU")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.xlim([-5,5])
    # plt.axis('equal')
    plt.savefig('./PICS/Approx_gelu.pdf')
