import numpy as np

def smoothness_derivative_energy(x, y, order=1):
    """
    x, y: 1D arrays of points
    order: 1 for first derivative energy, 2 for second derivative (bending energy)
    """
    y = np.array(y)
    if order == 1:
        dy = np.diff(y) / np.diff(x)
        return np.sum(dy**2)
    elif order == 2:
        dy = np.diff(y, 2) / np.diff(x[:-1])**2
        return np.sum(dy**2)
    else:
        raise ValueError("order must be 1 or 2")
    
def curvature_smoothness(x, y):
    """
    Returns total curvature as a smoothness measure.
    Lower total curvature → smoother.
    """
    x, y = np.array(x), np.array(y)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
    return np.sum(curvature)  # total curvature

def lipschitz_constant(x, y):
    """
    Approximate maximum slope between consecutive points
    """
    x, y = np.array(x), np.array(y)
    slopes = np.abs(np.diff(y) / np.diff(x))
    return np.max(slopes)

def frequency_smoothness(y, threshold=0.1):
    """
    y: 1D array of values
    threshold: fraction of high frequencies to sum
    Returns sum of high-frequency energy (lower → smoother)
    """
    y = np.array(y)
    n = len(y)
    f = np.fft.fft(y)
    power = np.abs(f)**2
    k = int(threshold * n)
    high_freq_energy = np.sum(power[k:])
    return high_freq_energy