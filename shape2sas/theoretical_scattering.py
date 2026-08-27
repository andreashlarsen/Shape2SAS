"""Theoretical scattering: pair distances, p(r), the form factor P(q)
and the resulting intensity I(q)."""

import os

import numpy as np
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 

from .helpfunctions import getStructureFactorClass, printt, sinc

def calc_all_dist_func(point_distribution):
    """
    Calculate unique pairwise distances between 3D points.
    Returns a 1D float32 array of length N*(N-1)/2.
    """
    x_in,y_in,z_in = np.concatenate(point_distribution.x),np.concatenate(point_distribution.y),np.concatenate(point_distribution.z)
    x = np.array(x_in).astype(np.float32, copy=False)
    y = np.array(y_in).astype(np.float32, copy=False)
    z = np.array(z_in).astype(np.float32, copy=False)
    N = len(x)

    dist = np.empty(N * (N - 1) // 2, dtype=np.float32)

    k = 0
    for i in range(N - 1):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dz = z[i] - z[i+1:]
        dist[k : k + (N - i - 1)] = np.sqrt(dx*dx + dy*dy + dz*dz)
        k += N - i - 1

    return dist

def calc_all_contrasts_func(point_distribution):
    """
    Calculate unique pairwise contrast products of p.
    Returns a 1D float32 array of length N*(N-1)/2,
    matching calc_all_dist().
    """

    sld_in = np.concatenate(point_distribution.sld)
    sld = np.array(sld_in).astype(np.float32, copy=False)
    N = len(sld)

    # Preallocate result array (unique pairs only)
    contrasts = np.empty(N * (N - 1) // 2, dtype=np.float32)

    # Fill it using triangular indexing without making an (N, N) array
    k = 0
    for i in range(N - 1):
        # multiply p[i] with all following elements at once
        contrasts[k : k + (N - i - 1)] = sld[i] * sld[i+1:]
        k += N - i - 1

    return contrasts
    
def generate_histogram_func(dist, prpoints, contrast, r_max):
    """
    make histogram of point pairs, h(r), binned after pair-distances, r
    used for calculating scattering (fast Debye)

    input
    dist     : all pairwise distances
    prpoints : number of bins in h(r)
    contrast : contrast of points
    r_max    : max distance to include in histogram

    output
    r        : distances of bins
    h    : histogram, weighted by contrast

    """

    h, bin_edges = np.histogram(dist, bins=prpoints, weights=contrast, range=(0,r_max)) 
    r = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    return r, h
    
def calc_hr_func(dist, prpoints, contrast, polydispersity):
    """
    calculate h(r)
    h(r) is the contrast-weighted histogram of distances, including self-terms (dist = 0)

    input: 
    dist      : all pairwise distances
    contrast  : all pair-wise contrast products
    polydispersity: relative polydispersity, float

    output:
    hr        : pair distance distribution function 
    """
    if dist.dtype != np.float32:
        dist = dist.astype(np.float32, copy=False)
    if contrast.dtype != np.float32:
        contrast = contrast.astype(np.float32, copy=False)

    ## make r range in h(r) histogram slightly larger than dmax
    ratio_rmax_dmax = 1.05

    lognormal = False
    ## calc h(r) with/without polydispersity
    if polydispersity > 0.0:
        if lognormal:
            dmax = np.amax(dist)*np.exp(3* polydispersity)
        else:
            dmax = np.amax(dist) * (1 + 3 * polydispersity)
        r_max = dmax * ratio_rmax_dmax
        r, hr_1 = generate_histogram_func(dist, prpoints, contrast, r_max)
        N_poly_integral = 25 # should be uneven to include 1 in factor_range (precalculated)
        hr  = np.zeros_like(hr_1, dtype=np.float32)
        #norm = 0.0
        if lognormal:
            log_factors = np.linspace(-3*polydispersity, 3*polydispersity, N_poly_integral, dtype=np.float32)
            factor_range = np.exp(log_factors)
        else:
            factor_range = 1 + np.linspace(-3, 3, N_poly_integral, dtype=np.float32) * polydispersity
        res_range = (1.0 - factor_range) / polydispersity
        if lognormal:
            w_range = np.exp(-(np.log(factor_range))**2 / (2*polydispersity**2)) / (factor_range * polydispersity * np.sqrt(2*np.pi))
        else:
            w_range = np.exp(-0.5*res_range**2)
        vol2_range = factor_range**6
        norm_range = w_range*vol2_range
        for i,factor_d in enumerate(factor_range):
            if factor_d == 1.0:
                hr += hr_1
                #norm += 1.0
            else:
                # calculate in the same bins so histograms can be added
                dhr = histogram1d(dist * factor_d, bins=prpoints, weights=contrast, range=(0,r_max))
                hr += dhr * norm_range[i]
        norm = np.sum(norm_range)
        hr /= norm
    else:
        dmax = np.amax(dist)
        r_max = dmax * ratio_rmax_dmax
        r, hr = generate_histogram_func(dist, prpoints, contrast, r_max)

    return r, hr, dmax

def calc_Rg_func(r, pr):
    """ 
    calculate Rg from r and p(r)
    """
    sum_pr_r2 = np.sum(pr * r**2)
    sum_pr = np.sum(pr)
    Rg = np.sqrt(abs(sum_pr_r2 / sum_pr) / 2)

    return Rg
    
def calc_pr_func(point_distribution,prpoints=100,polydispersity=0):
    """
    calculate p(r)
    p(r) is the contrast-weighted histogram of distances, without the self-terms (dist = 0)

    input: 
    dist      : all pairwise distances
    contrast  : all pair-wise contrast products
    polydispersity: boolian, True or False

    output:
    pr        : pair distance distribution function
    """
    printt('        calculating distances...')
    dist = calc_all_dist_func(point_distribution)
    printt('        calculating contrasts...')
    contrast = calc_all_contrasts_func(point_distribution)

    ## calculate pr
    printt('        calculating p(r)...')
    r, pr, dmax = calc_hr_func(dist, prpoints, contrast, polydispersity)
    printt(f"           dmax: {dmax:.3e} A")

    ## normalize so pr_max = 1
    pr_norm = pr / np.amax(pr)

    ## calculate Rg
    Rg = calc_Rg_func(r, pr_norm)
    printt(f"           Rg  : {Rg:.3e} A")

    #returned N values after generating
    pr /= len(point_distribution.x)**2 #NOTE: N_total**2

    return r, pr, pr_norm, dmax

def calc_Pq_func(q, r, pr, conc, volume_total):
    """
    calculate form factor, P(q), and forward scattering, I(0), using pair distribution, p(r) 
    """
    ## calculate P(q) and I(0) from p(r)
    I0, Pq = 0, 0
    for (r_i, pr_i) in zip(r, pr):
        I0 += pr_i
        qr = q * r_i
        Pq += pr_i * sinc(qr)

    # normalization, P(0) = 1
    if I0 == 0:
        I0 = 1E-5
    elif I0 < 0:
        I0 = abs(I0)
    Pq /= I0

    # make I0 scale with volume fraction (concentration) and 
    # volume squared and scale so default values gives I(0) of approx unity
    I0 *= conc * volume_total * 1E-4
    return I0, Pq

def calc_Iq_func(q, Pq, S_eff, sigma_r):
    """
    calculates intensity
    """

    ## multiply form factor with structure factor
    I = Pq * S_eff

    ## interface roughness (Skar-Gislinge et al. 2011, DOI: 10.1039/c0cp01074j)
    if sigma_r > 0.0:
        roughness = np.exp(-(q * sigma_r)**2 / 2)
        I *= roughness

    return I
    
def calc_S_func(q,point_distribution,stype,S_par,Pq):
    """calculate the effective structure factor S_eff(q)

    the structure factors themselves live in the structure_factors folder
    """
    return getStructureFactorClass(stype)(S_par).structure_eff(q,point_distribution,Pq)

def save_pr_func(r,pr,model_filename):
    """save pair distance distribution p(r)"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/pr_%s.dat' % (model_filename,model_filename),'w') as f:
        #f.write('# Pair distance distribution function (PDDF) p(r)\n')
        f.write('# %-12s %-12s\n' % ('r','p(r)'))
        for r_i,pr_i in zip(r,pr):
            f.write('  %-12.5e %-12.5e\n' % (r_i, pr_i))

def save_I_func(q, I, model_filename):
    """Save theoretical intensity to file"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/Iq_%s.dat' % (model_filename,model_filename),'w') as f:
        f.write('# Theoretical SAS data\n')
        f.write('# %-12s %-12s\n' % ('q','I'))
        for qi,Ii in zip(q,I):
            f.write('  %-12.5e %-12.5e\n' % (qi, Ii))

