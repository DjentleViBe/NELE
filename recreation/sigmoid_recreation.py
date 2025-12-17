# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Recreating SIGMOID with NELE
"""
import numpy as np

def sigmoid(x):
    """
    SIGMOID function
    
    :param x: X values
    """
    return 1 / (1 + np.exp(-x))
