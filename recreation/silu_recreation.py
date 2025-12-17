# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating SILU with NELE
"""
import numpy as np

def silu(x):
    """
    SILU function
    
    :param x: X values
    """
    return x / (1 + np.exp(-x))
