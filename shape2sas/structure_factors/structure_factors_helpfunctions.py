import numpy as np
from ..helpfunctions import sinc, calc_com_dist

def calc_A00(q,point_distribution):
    """
    calc zeroth order sph harm, for decoupling approximation
    """
    d = calc_com_dist(point_distribution).astype(np.float32, copy=False)
    M = len(q)
    A00 = np.zeros(M)
    sld = np.concatenate(point_distribution.sld).astype(np.float32, copy=False)
    for i in range(M):
        qr = q[i] * d
        A00[i] = np.sum(sld * sinc(qr))
    A00 = A00 / A00[0] # normalise, A00[0] = 1

    return A00

def decoupling_approx(q,point_distribution,Pq,S):
    """
    modify structure factor with the decoupling approximation
    for combining structure factors with non-spherical (or polydisperse) particles

    see, for example, Larsen et al 2020: https://doi.org/10.1107/S1600576720006500
    and refs therein

    input
    q
    point_distribution : coordinates and contrasts
    Pq                 : form factor
    S                  : structure factor

    output
    S_eff              : effective structure factor, after applying decoupl. approx
    """
    A00 = calc_A00(q,point_distribution)
    const = 1e-3 # add constant in nominator and denominator, for stability (numerical errors for small values dampened)
    Beta = (A00**2 + const) / (Pq + const)
    S_eff = 1 + Beta * (S - 1)
    return S_eff
