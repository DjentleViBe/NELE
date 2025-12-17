import numpy as np
from recreation.elu_recreation import nurbs_gen_elu, elu, loss_elu
from recreation.silu_recreation import nurbs_gen_silu, silu, loss_silu
from recreation.lelu_recreation import nurbs_gen_lelu, lelu, loss_lelu
from recreation.mish_recreation import nurbs_gen_mish, mish, loss_mish
from recreation.sigmoid_recreation import nurbs_gen_sigmoid, sigmoid, loss_sigmoid
from recreation.tanh_recreation import nurbs_gen_tanh, tanh, loss_tanh
from recreation.softplus_recreation import nurbs_gen_softplus, softplus, loss_softplus
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import config as cfg

typearray = cfg.curve_recreation
type = 'ELU'
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
    'elu': loss_elu,
    'silu': loss_silu,
    'lelu': loss_lelu,
    'mish': loss_mish,
    'sigmoid': loss_sigmoid,
    'tanh': loss_tanh,
    'softplus': loss_softplus,
    # add more…
}
curve_functions = {
    'elu': nurbs_gen_elu,
    'silu': nurbs_gen_silu,
    'lelu': nurbs_gen_lelu,
    'mish': nurbs_gen_mish,
    'sigmoid': nurbs_gen_sigmoid,
    'tanh': nurbs_gen_tanh,
    'softplus': nurbs_gen_softplus,
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
for type in typearray:
    loss_fn = loss_functions[type.lower()]
    target_fn = target_functions[type.lower()]
    res = minimize(
        loss_fn,
        init_params,
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

    cur = curve_functions[type.lower()]
    curve = cur(opt_ctrl, opt_w, t)
    plt.figure(figsize=(5, 4))
    plt.plot(np.concatenate([x_target, x_extra]), np.concatenate([y_target, y_extra]), '--', label=r"$\texttt{ELU}$", color = 'red')
    plt.rcParams['text.usetex'] = True
    plt.plot(curve[:, 0], curve[:, 1], label = r'$\texttt{ELU}$ approximation', color = 'k', linewidth=0.7)
    plt.scatter(opt_ctrl[:, 0], opt_ctrl[:, 1], label=r'Control Points $\texttt{NELE}$', color = 'k', s=3)
    print(f'\\texttt{{{type}}} & [{round(opt_ctrl[0][0], 3)}, {round(opt_ctrl[0][1], 3)}] &\
    [{round(opt_ctrl[1][0], 3)}, {round(opt_ctrl[1][1], 3)}] &\
    [{round(opt_ctrl[2][0], 3)}, {round(opt_ctrl[2][1], 3)}] &\
    [{round(opt_ctrl[3][0], 3)}, {round(opt_ctrl[3][1], 3)}] &\
    [{round(opt_w[0], 3)} , {round(opt_w[1], 3)} , {round(opt_w[2], 3)} , {round(opt_w[3], 3)}]\\\\')
    plt.legend()
    plt.grid(True, linewidth = 0.2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig('./PICS/Approx'+ type +'.pdf')
