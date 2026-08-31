"""SESANS: the projected correlation function G(delta) and its simulated data."""

import numpy as np
from scipy.special import j0

def calc_G_sesans(q,delta,I):
    """
    Calculated projected correlation function for SESANS from Hankel Transform of I(q)
    """

    # Init empty G(delta)
    G = np.empty(len(delta), dtype=float)

    # calculate G(delta) from I(q)
    for i, delta_i in enumerate(delta):
        dq_int = q[1] - q[0]
        G[i] = 1 / 2 / np.pi * np.sum(dq_int * q * I * j0(delta_i * q))

    return G

def simulate_sesans(delta,G,error):
    """
    Simulate SESANS data using calculated scattering and estimate for sigma

    input
    delta, G: spin-echo lengths and theoretical G(delta)
    error: relative error

    output
    sesans_sigma: simulated errors
    lnPsim: simulated data

    """
    # Compute baseline noise as sesans_noise % of min(G-G(0))
    noise_baseline = error * np.abs(np.min(G - G[0]))
    # Compute delta-dependent noise as function of baseline noise
    m = 1/50000 # 1/50000 adds a baseline worth of noise per 5 micrometers of additional spin echo length (delta)
    d_delta = delta[-1] - delta[0]
    sesans_sigma = np.linspace(noise_baseline, noise_baseline * (1 + m * d_delta), len(delta))
    # pick random points using mean and sigma
    lnPsim = np.random.normal((G - G[0]), sesans_sigma)
    return lnPsim,sesans_sigma

