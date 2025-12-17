# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating SOFTPLUS with NELE
"""
import numpy as np

def softplus(x):
    """
    SOFTPLUS function
    
    :param x: X values
    """
    return np.log1p(np.exp(x))
