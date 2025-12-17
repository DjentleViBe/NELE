# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating MISH with NELE
"""
import numpy as np

def mish(x):
    """
    MISHfunction
    
    :param x: X values
    """
    return x * np.tanh(np.log1p(np.exp(x)))
