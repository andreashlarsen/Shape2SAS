"""Simulated experimental data: noise added to the theoretical intensity."""

import os

import numpy as np

def simulate_data_func(q,I,I0,exposure):
    """
    Simulate SAXS data using calculated scattering and empirical expression for sigma
    using Sedlak et al, 2017: (https://doi.org/10.1107/S1600576717003077)

    input
    q,I      : calculated scattering, normalized
    I0       : forward scattering
    exposure : exposure (in arbitrary units) - affects the noise level of data

    output
    sigma    : simulated noise
    Isim     : simulated data

    data is also written to a file
    """

    # set constants
    k = 4500
    c = 0.85

    # convert from intensity units to counts
    I_sed = exposure * I0 * I

    # make N
    N = k * q # original expression from Sedlak2017 paper

    qt = 1.4 # threshold - above this q value, the linear expression do not hold
    a = 3.0 # empirical constant 
    b = 0.6 # empirical constant
    idx = np.where(q > qt)
    N[idx] = k * qt * np.exp(-0.5 * ((q[idx] - qt) / b)**a)

    # make I(q_arb)
    q_max = np.amax(q)
    q_arb = 0.3
    if q_max <= q_arb:
        I_sed_arb = I_sed[-2]
    else: 
        idx_arb = np.where(q > q_arb)[0][0]
        I_sed_arb = I_sed[idx_arb]

    # calc variance and sigma
    v_sed = (I_sed + 2 * c * I_sed_arb / (1 - c)) / N
    sigma_sed = np.sqrt(v_sed)

    # rescale
    sigma = sigma_sed / exposure

    ## simulate data using errors
    mu = I0 * I
    Isim = np.random.normal(mu, sigma)

    return Isim, sigma

def save_Isim_func(q, I_sim, sigma, model_filename):
    """Save theoretical intensity to file"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/Isim_%s.dat' % (model_filename,model_filename),'w') as f:
        f.write('# Simulated SAXS data with noise\n')
        f.write('# sigma generated using Sedlak et al, k=100000, c=0.55, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n')
        f.write('# %-12s %-12s %-12s\n' % ('q','I','sigma'))
        for q_i,Isim_i,sigma_i in zip(q,I_sim,sigma):
                f.write('  %-12.5e %-12.5e %-12.5e\n' % (q_i, Isim_i, sigma_i))

