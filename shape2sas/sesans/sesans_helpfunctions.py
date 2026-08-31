import numpy as np

def default_sesans_range(dmax):
    """default spin echo length range, set by the size of the particle itself"""
    qmin = 0.001 * np.pi / dmax
    deltamax = 3 * dmax
    return qmin, deltamax