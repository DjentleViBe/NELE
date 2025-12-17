# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating LELU with NELE
"""
import numpy as np

def lelu(x, beta=0.1):
    """
    LELU function
    
    :param x: X values
    """
    mask = x >= 0
    return np.where(mask, x, np.exp((1 - beta) * x) - 1 + beta * x)
