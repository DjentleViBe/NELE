# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating ELU with NELE
"""
import numpy as np

def elu(x):
    """
    ELU function
    
    :param x: X values
    """
    return np.where(x >= 0, x, np.exp(x) - 1)
