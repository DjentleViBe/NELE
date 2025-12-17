# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating GELU with NELE
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from recreation.nurbs_gen import nurbs_gen
from recreation.gen_loss import loss_af

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

    opt_ctrl_mnist = np.array([
        [-4, 0],
        [-0.1, -0.1],
        [-1/2**0.5, -1/2**0.5],
        [0, 0]
    ])
    opt_w_mnist = np.array([1.0, 1.0, 1.0, 1.0])
    linear = np.array([[0, 0], [4, 4]])
    linear_2 = np.array([[-8, 0], [-4, 0]])
    # ---------------------------
    # Plot results
    # ---------------------------
    curve_gen = nurbs_gen(opt_ctrl, opt_w, t_val)
    curve_mnist = nurbs_gen(opt_ctrl_mnist, opt_w_mnist, t_val)
    plt.figure(figsize=(5, 4))
    plt.rcParams['text.usetex'] = True
    plt.plot(np.concatenate([x_target, x_extra]), np.concatenate([y_target, y_extra]), \
            '--', label=r'$\texttt{GELU}$', color = 'red')
    plt.plot(curve_gen[:, 0], curve_gen[:, 1], label = r'$\texttt{GELU}$ approximation', \
            color = 'k', linewidth=0.7)
    plt.plot(curve_mnist[:, 0], curve_mnist[:, 1], label=r'$\texttt{NELE}$ MNIST', \
            color = 'green', linewidth=1.0)
    plt.scatter(opt_ctrl[:, 0], opt_ctrl[:, 1], label=r'Control Points $\texttt{NELE}$', \
                color = 'k', s=3)
    plt.scatter(opt_ctrl_mnist[:, 0], opt_ctrl_mnist[:, 1], label="Control Points MNIST", \
                color = 'green', marker = '+')
    # Label each point p0, p1, p2, p3
    for i, (x_val, y) in enumerate(opt_ctrl_mnist):
        plt.annotate(f"p{i}", (x_val, y), textcoords="offset points", \
                    xytext=(5, -4), fontsize=6, color='green')
    plt.plot(linear[:, 0], linear[:, 1], color = 'green', linewidth=1.0)
    plt.plot(linear_2[:, 0], linear_2[:, 1], color = 'green', linewidth=1.0)

    plt.legend()
    plt.grid(True, linewidth = 0.2)
    #plt.title("Optimized Cubic NURBS Approximation of GELU")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.xlim([-5,5])
    # plt.axis('equal')
    plt.savefig('./PICS/Approx_gelu.pdf')
