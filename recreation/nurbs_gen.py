# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Generate nurbs
"""
def nurbs_gen(control_points, weights, t):
    """
    Generate ELU nurbs
    
    :param control_points: Control points values
    :param weights: weights values
    :param t: parameter space
    """
    cp = control_points.reshape(4, 2)
    w = weights

    n0_val = (1 - t)**3
    n1_val = 3 * t * (1 - t)**2
    n2_val = 3 * t**2 * (1 - t)
    n3_val = t**3

    numerator = (n0_val[:, None] * w[0] * cp[0] +
                 n1_val[:, None] * w[1] * cp[1] +
                 n2_val[:, None] * w[2] * cp[2] +
                 n3_val[:, None] * w[3] * cp[3])

    denominator = (n0_val * w[0] +
                   n1_val * w[1] +
                   n2_val * w[2] +
                   n3_val * w[3])[:, None]

    return numerator / (denominator + 1e-12)
