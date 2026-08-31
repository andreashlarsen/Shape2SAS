import os

import numpy as np
from ..helpfunctions import printt, sinc, calc_com_dist

def check_Spar(name,S_par,par_names):
    """check if the number of input parameters for the structure factor is correct, else return error message"""
    n = len(par_names)
    len_par = len(S_par)
    if len_par != n:
        par = ' parameter ' if n == 1 else ' parameters '
        were = ' was ' if len_par == 1 else ' were '
        printt("\nERROR: structure factor " + name + " needs " + str(n) + par + "(provided after --S_par or -Sp): " + ", ".join(par_names) + ", but " + str(len_par) + were + "given: " + str(list(S_par)) + "\n")
        exit()

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

def save_S_func(q, S, model_filename):
    """Save structure factor to file"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/Sq_%s.dat' % (model_filename,model_filename),'w') as f:
        f.write('# Structure factor SAS data\n')
        f.write('# %-12s %-12s\n' % ('q','S'))
        for qi,Si in zip(q,S):
            f.write('  %-12.5e %-12.5e\n' % (qi, Si))

