# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Activation function revreation with NELE
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import config as cfg
from recreation.nurbs_gen import nurbs_gen
from recreation.elu_recreation import elu
from recreation.silu_recreation import silu
from recreation.lelu_recreation import lelu
from recreation.mish_recreation import mish
from recreation.sigmoid_recreation import sigmoid
from recreation.tanh_recreation import tanh
from recreation.softplus_recreation import softplus
from recreation.gen_loss import loss_af
from recreation.gelu_recreation import gelu_special

typearray = cfg.curve_recreation
STUDY_TYPE = 'ELU'
np.set_printoptions(precision=4, suppress=True)
initial_ctrl = np.array([
    [-4, 0],
    [-2.2, -0.02],
    [-0.7, -0.08],
    [0, 0]
]).flatten()

initial_weights = np.array([1.0, 0.9, 0.7, 1.0])
init_params = np.concatenate([initial_ctrl, initial_weights])
loss_functions = {
    'elu': loss_af,
    'silu': loss_af,
    'lelu': loss_af,
    'mish': loss_af,
    'sigmoid': loss_af,
    'tanh': loss_af,
    'softplus': loss_af,
    # add more…
}
curve_functions = {
    'elu': nurbs_gen,
    'silu': nurbs_gen,
    'lelu': nurbs_gen,
    'mish': nurbs_gen,
    'sigmoid': nurbs_gen,
    'tanh': nurbs_gen,
    'softplus': nurbs_gen,
    # add more…
}
target_functions = {
    'elu': elu,
    'silu': silu,
    'lelu': lelu,
    'mish': mish,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softplus': softplus,
    # add more…
}
x_target = np.linspace(-4, 0, 200)

for studytype in typearray:
    loss_fn = loss_functions[studytype.lower()]
    target_fn = target_functions[studytype.lower()]
    y_target = target_functions[studytype.lower()](x_target)
    t_val = (x_target - x_target.min()) / (x_target.max() - x_target.min())
    res = minimize(
        loss_fn,
        init_params,
        args=(t_val, x_target, y_target),
        method='L-BFGS-B',
        options={'maxiter': 500}
    )
    # target segment
    x_target = np.linspace(-4, 0, 200)
    y_target = target_fn(x_target)

    x_extra = np.linspace(0, 4, 200)
    y_extra = target_fn(x_extra)

    # reparameterize into t ∈ [0, 1]
    t = (x_target - x_target.min()) / (x_target.max() - x_target.min())

    opt_params = res.x
    opt_ctrl = opt_params[:8].reshape(4, 2)
    opt_w = opt_params[8:]

    cur = curve_functions[studytype.lower()]
    curve = cur(opt_ctrl, opt_w, t)
    plt.figure(figsize=(5, 4))
    plt.plot(np.concatenate([x_target, x_extra]), \
             np.concatenate([y_target, y_extra]), '--', \
                label=r"$\texttt{ELU}$", color = 'red')
    plt.rcParams['text.usetex'] = True
    plt.plot(curve[:, 0], curve[:, 1],\
             label = r'$\texttt{ELU}$ approximation', \
                color = 'k', linewidth=0.7)
    plt.scatter(opt_ctrl[:, 0], opt_ctrl[:, 1], \
                label=r'Control Points $\texttt{NELE}$', \
                color = 'k', s=3)
    print(f'\\texttt{{{studytype}}} & [{round(opt_ctrl[0][0], 3)}, {round(opt_ctrl[0][1], 3)}] &\
    [{round(opt_ctrl[1][0], 3)}, {round(opt_ctrl[1][1], 3)}] &\
    [{round(opt_ctrl[2][0], 3)}, {round(opt_ctrl[2][1], 3)}] &\
    [{round(opt_ctrl[3][0], 3)}, {round(opt_ctrl[3][1], 3)}] &\
    [{round(opt_w[0], 3)} , {round(opt_w[1], 3)} ,\
          {round(opt_w[2], 3)} , {round(opt_w[3], 3)}]\\\\')
    plt.legend()
    plt.grid(True, linewidth = 0.2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig('./PICS/Approx_'+ studytype.lower() +'.pdf')

gelu_special()
