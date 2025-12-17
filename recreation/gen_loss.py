# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Generic loss function
"""
import numpy as np
from recreation.nurbs_gen import nurbs_gen

def loss_af(params, t_val, x_target, y_target):
    """
    Loss functon
    
    :param params: Parmeters used for the optimisation
    """
    # params = 4 ctrl points * 2 coords + 4 weights = 12 values
    ctrl = params[:8].reshape(4, 2)
    w = params[8:]

    curve = nurbs_gen(ctrl, w, t_val)

    # match x and y separately
    x_cur, y_cur = curve[:, 0], curve[:, 1]

    # target x is known: linear from -4 to 0
    x_desired = x_target

    return np.mean((x_cur - x_desired)**2 + (y_cur - y_target)**2)
