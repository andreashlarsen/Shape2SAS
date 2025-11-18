import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 
import inspect
import sys
import re
import warnings
import os
from dataclasses import dataclass
#from subunits import Sphere,Cylinder
from subunits import *

def printt(s): 
    """ print and write to log file"""
    print(s)
    with open('shape2sas.log','a') as f:
        f.write('%s\n' %s)

def sinc(x):
    """
    function for calculating sinc = sin(x)/x
    numpy.sinc is defined as sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.sinc(x / np.pi)   

def get_header_footer(file):
    """Count header and footer lines with non-numeric entries."""
    header, footer = 0, 0
    with open(file, errors='ignore') as f:
        lines = f.readlines()

    CONTINUE_H, CONTINUE_F = True, True
    j = 0
    while CONTINUE_H or CONTINUE_F:
        line_h, line_f = lines[j], lines[-1-j]
        tmp_h, tmp_f = line_h.split(), line_f.split()
        if CONTINUE_H:
            try:
                for val in tmp_h[:3]:
                    float(val)
                CONTINUE_H = False
            except:
                header += 1
        if CONTINUE_F:
            try:
                for val in tmp_f[:3]:
                    float(val)
                CONTINUE_F = False
            except:
                footer += 1
        j += 1
    return header, footer

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
    
def calc_com_dist_func(point_distribution):
    """ 
    calc contrast-weighted com distance
    """
    w = np.abs(point_distribution.sld)

    if np.sum(w) == 0:
        w = np.ones(len(point_distribution.x))

    x_com, y_com, z_com = np.average(point_distribution.x, weights=w), np.average(point_distribution.y, weights=w), np.average(point_distribution.z, weights=w)
    dx, dy, dz = point_distribution.x - x_com, point_distribution.y - y_com, point_distribution.z - z_com
    com_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    return com_dist

def calc_A00_func(q,point_distribution):
    """
    calc zeroth order sph harm, for decoupling approximation
    """
    d_in = np.concatenate(calc_com_dist_func(point_distribution))
    d = np.array(d_in).astype(np.float32, copy=False)
    M = len(q)
    A00 = np.zeros(M)
    sld_in = np.concatenate(point_distribution.sld)
    sld = np.array(sld_in).astype(np.float32, copy=False)
    for i in range(M):
        qr = q[i] * d
        A00[i] = sum(sld * sinc(qr))
    A00 = A00 / A00[0] # normalise, A00[0] = 1

    return A00

def decoupling_approx_func(q,point_distribution, Pq, S):
    """
    modify structure factor with the decoupling approximation
    for combining structure factors with non-spherical (or polydisperse) particles

    see, for example, Larsen et al 2020: https://doi.org/10.1107/S1600576720006500
    and refs therein

    input
    q
    x,y,z,p    : coordinates and contrasts
    Pq         : form factor
    S          : structure factor

    output
    S_eff      : effective structure factor, after applying decoupl. approx
    """
    A00 = calc_A00_func(q,point_distribution)
    const = 1e-3 # add constant in nominator and denominator, for stability (numerical errors for small values dampened)
    Beta = (A00**2 + const) / (Pq + const)
    S_eff = 1 + Beta * (S - 1)
    return S_eff

def calc_S_HS_func(q,conc,R_HS):
    """
    Calculate the hard-sphere structure factor using the Percus-Yevick approximation.
    Implements the stable version with Taylor expansion for small A = 2*R*q.
    adapted directly from the sasview code
    """

    if conc <= 0.0:
        return np.ones(len(q))

    vf = conc
    R = R_HS
    X = np.abs(2.0 * R * q)  # same as A in your earlier code

    # Precompute constants
    denom = (1.0 - vf)
    if denom < 1e-12:  # avoid division by zero
        return np.ones_like(q)

    Xinv = 1.0 / denom
    D = Xinv * Xinv
    A = (1.0 + 2.0 * vf) * D
    A *= A
    B = (1.0 + 0.5 * vf) * D
    B *= B
    B *= -6.0 * vf
    G = 0.5 * vf * A

    # Cutoffs
    cutoff_tiny = 5e-6
    cutoff_series = 0.05  # corresponds to CUTOFFHS in C code

    S_HS = np.empty_like(q)

    for i, x in enumerate(X):
        if x < cutoff_tiny:
            # limit q -> 0
            S_HS[i] = 1.0 / A
        elif x < cutoff_series:
            # Taylor series expansion
            x2 = x * x
            # Equivalent to the FF expression in the C code
            FF = (8.0 * A + 6.0 * B + 4.0 * G
                + (-0.8 * A - B / 1.5 - 0.5 * G
                    + (A / 35.0 + 0.0125 * B + 0.02 * G) * x2) * x2)
            S_HS[i] = 1.0 / (1.0 + vf * FF)
        else:
            # Normal expression
            x2 = x * x
            x4 = x2 * x2
            s, c = np.sin(x), np.cos(x)
            # FF expression refactored from the C code
            FF = ((G * ((4.0 * x2 - 24.0) * x * s
                        - (x4 - 12.0 * x2 + 24.0) * c
                        + 24.0) / x2
                + B * (2.0 * x * s - (x2 - 2.0) * c - 2.0)) / x
                + A * (s - x * c)) / x
            S_HS[i] = 1.0 / (1.0 + 24.0 * vf * FF / x2)

    return S_HS

def calc_S_aggr_func(q,Reff,Naggr):
    """
    calculates fractal aggregate structure factor with dimensionality 2

    S_{2,D=2} in Larsen et al 2020, https://doi.org/10.1107/S1600576720006500

    input 
    q      :
    Naggr  : number of particles per aggregate
    Reff   : effective radius of one particle 

    output
    S_aggr :
    """
    qR = q * Reff
    S_aggr = 1 + (Naggr - 1)/(1 + qR**2 * Naggr / 3)
    return S_aggr

def structure_eff(self, Pq):
    """Return effective structure factor for aggregation"""

    S = self.calc_S_aggr()
    S_eff = self.decoupling_approx(Pq, S)
    S_eff = (1 - self.fracs_aggr) + self.fracs_aggr * S_eff
    return S_eff

def check_Spar(stype,S_par,n):
    if len(S_par) != n:
            if n == 1:
                printt("\nERROR: structure factor " + stype + " needs " + str(n) + " parameter (provided after --S_par or -Sp), but " + str(len(S_par)) + ' parameters were given: ' + str(S_par) + '\n')
            else:
                printt("\nERROR: structure factor " + stype + " needs " + str(n) + " parameters (provided after --S_par or -Sp), but " + str(len(S_par)) + ' parameters were given: ' + str(S_par) + '\n')
            exit()

def calc_S_func(q,point_distribution,stype,S_par,Pq):
    aliasses_HS = ['hardsphere','hs','hard-sphere']
    aliasses_aggr = ['aggregation','aggr','aggregate','frac2d']

    if stype in aliasses_HS:
        check_Spar(stype,S_par,2)
        conc,R_HS = S_par
        S = calc_S_HS_func(q,conc,R_HS)
        S_eff = decoupling_approx_func(q,point_distribution,Pq, S)
    elif stype in aliasses_aggr:
        check_Spar(stype,S_par,3)
        Reff,Naggr,fracs_aggr = S_par
        S = calc_S_aggr_func(q,Reff,Naggr)
        S_eff = decoupling_approx_func(q,point_distribution,Pq, S)
        S_eff = (1 - fracs_aggr) + fracs_aggr * S_eff
    else:
        S_eff = np.ones_like(q)
    
    return S_eff

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

def save_points(point_distribution,model_filename):
    """save point cloud to a file"""
    os.makedirs(model_filename, exist_ok=True)  
    x,y,z,sld = np.concatenate(point_distribution.x), np.concatenate(point_distribution.y), np.concatenate(point_distribution.z), np.concatenate(point_distribution.sld)
    with open('%s/points_%s.txt' % (model_filename,model_filename),'w') as f:
        f.write('# x y z sld\n')
        for xi,yi,zi,s in zip(x,y,z,sld):
            f.write('%f %f %f %f\n' % (xi,yi,zi,s))

def save_pr_func(r,pr,model_filename):
    """save pair distance distribution p(r)"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/pr_%s.dat' % (model_filename,model_filename),'w') as f:
        #f.write('# Pair distance distribution function (PDDF) p(r)\n')
        f.write('# %-12s %-12s\n' % ('r','p(r)'))
        for r_i,pr_i in zip(r,pr):
            f.write('  %-12.5e %-12.5e\n' % (r_i, pr_i))

def save_S_func(q, S, model_filename):
    """Save structure factor to file"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/Sq_%s.dat' % (model_filename,model_filename),'w') as f:
        f.write('# Structure factor SAS data\n')
        f.write('# %-12s %-12s\n' % ('q','S'))
        for qi,Si in zip(q,S):
            f.write('  %-12.5e %-12.5e\n' % (qi, Si))

def save_I_func(q, I, model_filename):
    """Save theoretical intensity to file"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/Iq_%s.dat' % (model_filename,model_filename),'w') as f:
        f.write('# Theoretical SAS data\n')
        f.write('# %-12s %-12s\n' % ('q','I'))
        for qi,Ii in zip(q,I):
            f.write('  %-12.5e %-12.5e\n' % (qi, Ii))

def save_Isim_func(q, I_sim, sigma, model_filename):
    """Save theoretical intensity to file"""

    os.makedirs(model_filename, exist_ok=True)  
    with open('%s/Isim_%s.dat' % (model_filename,model_filename),'w') as f:
        f.write('# Simulated SAXS data with noise\n')
        f.write('# sigma generated using Sedlak et al, k=100000, c=0.55, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)\n')
        f.write('# %-12s %-12s %-12s\n' % ('q','I','sigma'))
        for q_i,Isim_i,sigma_i in zip(q,I_sim,sigma):
                f.write('  %-12.5e %-12.5e %-12.5e\n' % (q_i, Isim_i, sigma_i))

def Rotate(x,y,z,alpha,beta,gamma):
    """
    Simple Euler rotation
    input angles in degrees
    """
    a,b,g = np.radians(alpha),np.radians(beta),np.radians(gamma)
    ca,cb,cg = np.cos(a),np.cos(b),np.cos(g)
    sa,sb,sg = np.sin(a),np.sin(b),np.sin(g)
    x_rot = ( x * cg * cb + y * (cg * sb * sa - sg * ca) + z * (cg * sb * ca + sg * sa))
    y_rot = ( x * sg * cb + y * (sg * sb * sa + cg * ca) + z * (sg * sb * ca - cg * sa))
    z_rot = (-x * sb      + y * cb * sa                  + z * cb * ca)
    return x_rot, y_rot, z_rot

class GenerateAllPoints:
    def __init__(self, Npoints, com, subunits, dimensions, rotation, sld, exclude_overlap):
        self.Npoints = Npoints
        self.com = com
        self.subunits = subunits
        self.Number_of_subunits = len(subunits)
        self.dimensions = dimensions
        self.rotation = rotation
        self.sld = sld
        self.exclude_overlap = exclude_overlap
        self.setAvailableSubunits()

    def setAvailableSubunits(self):
        """Dynamically build dictionary of aliases -> subunit classes"""
        current_module = sys.modules[__name__]
        classes = inspect.getmembers(current_module, inspect.isclass)
        self.subunitClasses = {}
        for _, cls in classes:
            if hasattr(cls, "aliases"):
                for alias in cls.aliases:
                    self.subunitClasses[alias.lower().replace("_", "").replace(" ", "")] = cls

    @staticmethod
    def AppendingPoints(x_new, y_new, z_new,sld_new, x_add, y_add, z_add, sld_add):
        """append new points to vectors of point coordinates"""
        
        # add points to (x_new,y_new,z_new)
        if isinstance(x_new, int):
            # if these are the first points to append to (x_new,y_new,z_new)
            x_new = x_add
            y_new = y_add
            z_new = z_add
            sld_new = sld_add
        else:
            x_new = np.append(x_new, x_add)
            y_new = np.append(y_new, y_add)
            z_new = np.append(z_new, z_add)
            sld_new = np.append(sld_new, sld_add)

        return x_new, y_new, z_new, sld_new

    @staticmethod
    def onCheckOverlap(x, y, z, p, rotation, com, subunitClass, dimensions):
        """
        check for overlap with previous subunits. 
        if overlap, the point is removed
        """
        # shift back to origin
        x_eff,y_eff,z_eff = x-com[0],y-com[1],z-com[2]
        if sum(rotation) != 0:
            #rotate back to original orientation
            alpha, beta, gamma = rotation
            x_eff, y_eff, z_eff  = Rotate(x_eff,y_eff,z_eff,-alpha,-beta,-gamma)

        # then check overlaps
        idx = subunitClass(dimensions).checkOverlap(x_eff, y_eff, z_eff)
        x_add, y_add, z_add, sld_add = x[idx], y[idx], z[idx], p[idx]

        ## number of excluded points
        N_x = len(x) - len(idx[0])
        return x_add, y_add, z_add, sld_add, N_x

    def onGeneratingAllPointsSeparately(self):
        """Generating points for all subunits from each built model, but
        save them separately in their own list"""
        volume = []
        sum_vol = 0

        #Get volume of each subunit
        for i in range(self.Number_of_subunits):

            subunitClass = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")]
            v = subunitClass(self.dimensions[i]).getVolume()
            volume.append(v)
            sum_vol += v

        N, rho, N_exclude = [], [], []
        x_new, y_new, z_new, sld_new, volume_total = [], [], [], [], 0

        for i in range(self.Number_of_subunits):
            Npoints = int(self.Npoints * volume[i] / sum_vol)
            
            x_add, y_add, z_add = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")](self.dimensions[i]).getPointDistribution(Npoints)
            alpha, beta, gamma = self.rotation[i]
            com_x, com_y, com_z = self.com[i]

            # rotate and translate
            x_add, y_add, z_add = Rotate(x_add, y_add, z_add,alpha,beta,gamma)
            x_add, y_add, z_add = x_add+com_x,y_add+com_y,z_add+com_z
            
            #Remaining points
            N_subunit = len(x_add)
            rho_subunit = N_subunit / volume[i]
            sld_add = np.ones(N_subunit) * self.sld[i]

            #Check for overlap with previous subunits
            N_x_sum = 0
            if self.exclude_overlap:
                for j in range(i): 
                    x_add, y_add, z_add, sld_add, N_x = self.onCheckOverlap(x_add, y_add, z_add, sld_add, self.rotation[j],  
                                                    self.com[j], self.subunitClasses[self.subunits[j].lower().replace("_", "").replace(" ", "")], self.dimensions[j])
                    N_x_sum += N_x
    
            N.append(N_subunit)
            rho.append(rho_subunit)
            N_exclude.append(N_x_sum)
            fraction_left = (N_subunit-N_x_sum) / N_subunit
            volume_total += volume[i] * fraction_left

            x_new.append(x_add)
            y_new.append(y_add)
            z_new.append(z_add)
            sld_new.append(sld_add)
        
        #Show information about the model and its subunits
        N_remain = []
        for j in range(self.Number_of_subunits):
            srho = rho[j] * self.sld[j]
            N_remain.append(N[j] - N_exclude[j])
            printt(f"        {N[j]} points for subunit {j}: {self.subunits[j]}")
            printt(f"             Point density     : {rho[j]:.3e} (points per volume)")
            printt(f"             Scattering density: {srho:.3e} (density times scattering length)")
            if self.exclude_overlap:
                printt(f"             Excluded points   : {N_exclude[j]} (overlap region)")
            else:
                printt(f"             Excluded points   : none - exclude overlap disabled")
            printt(f"             Remaining points  : {N_remain[j]} (non-overlapping region)")
        N_total = sum(N_remain)
        printt(f"        Total points in model: {N_total}")
        printt(f"        Total volume of model: {volume_total:.3e} A^3")
        printt(" ")

        return x_new, y_new, z_new, sld_new, volume_total

    def onGeneratingAllPoints(self):
        """Generating points for all subunits from each built model"""
        volume = []
        sum_vol = 0
        #Get volume of each subunit
        for i in range(self.Number_of_subunits):
            subunitClass = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")]
            v = subunitClass(self.dimensions[i]).getVolume()
            volume.append(v)
            sum_vol += v
        
        N, rho, N_exclude = [], [], []
        x_new, y_new, z_new, sld_new, volume_total = 0, 0, 0, 0, 0

        #Generate subunits
        for i in range(self.Number_of_subunits):
            Npoints = int(self.Npoints * volume[i] / sum_vol)
            
            x_add, y_add, z_add = self.subunitClasses[self.subunits[i].lower().replace("_", "").replace(" ", "")](self.dimensions[i]).getPointDistribution(self.Npoints)
            alpha, beta, gamma = self.rotation[i]
            com_x, com_y, com_z = self.com[i]

            # rotate and translate
            x_add, y_add, z_add = Rotate(x_add, y_add, z_add,alpha,beta,gamma)
            x_add, y_add, z_add = x_add+com_x,y_add+com_y,z_add+com_z

            #Remaining points
            N_subunit = len(x_add)
            rho_subunit = N_subunit / volume[i]
            sld_add = np.ones(N_subunit) * self.sld[i]

            #Check for overlap with previous subunits
            N_x_sum = 0
            if self.exclude_overlap:
                for j in range(i): 
                    x_add, y_add, z_add, sld_add, N_x = self.onCheckOverlap(x_add, y_add, z_add, sld_add, self.rotation[j],  
                                                    self.com[j], self.subunitClasses[self.subunits[j].lower().replace("_", "").replace(" ", "")], self.dimensions[j])
                    N_x_sum += N_x
            
            #Append points
            x_new, y_new, z_new, sld_new = self.AppendingPoints(x_new, y_new, z_new, sld_new, x_add, y_add, z_add, sld_add)

            N.append(N_subunit)
            rho.append(rho_subunit)
            N_exclude.append(N_x_sum)
            fraction_left = (N_subunit-N_x_sum) / N_subunit
            volume_total += volume[i] * fraction_left

        #Show information about the model and its subunits
        N_remain = []
        for j in range(self.Number_of_subunits):
            srho = rho[j] * self.sld[j]
            N_remain.append(N[j] - N_exclude[j])
            printt(f"        {N[j]} points for subunit {j}: {self.subunits[j]}")
            printt(f"             Point density     : {rho[j]:.3e} (points per volume)")
            printt(f"             Scattering density: {srho:.3e} (density times scattering length)")
            printt(f"             Excluded points   : {N_exclude[j]} (overlap region)")
            printt(f"             Remaining points  : {N_remain[j]} (non-overlapping region)")

        N_total = sum(N_remain)
        printt(f"        Total points in model: {N_total}")
        printt(f"        Total volume of model: {volume_total:.3e} A^3")
        printt(" ")

        return x_new, y_new, z_new, sld_new, volume_total

def get_max_dimension(x_list, y_list, z_list):
    """
    find max dimensions of n models
    used for determining plot limits
    """

    max_x,max_y,max_z = 0, 0, 0
    for i in range(len(x_list)):
        tmp_x = np.amax(abs(x_list[i]))
        tmp_y = np.amax(abs(y_list[i]))
        tmp_z = np.amax(abs(z_list[i]))
        if tmp_x>max_x:
            max_x = tmp_x
        if tmp_y>max_y:
            max_y = tmp_y
        if tmp_z>max_z:
            max_z = tmp_z

    max_l = np.amax([max_x,max_y,max_z])

    return max_l

def plot_2D(x_list, y_list, z_list, sld_list, model_filename_list, filetype, colors):
    """
    plot 2D-projections of generated points (shapes):
    positive contrast in red (Model 1) or blue (Model 2) or yellow (Model 3) or green (Model 4)
    zero contrast in grey
    negative contrast in black

    input
    (x_list,y_list,z_list) : coordinates of simulated points
    sld_list               : excess scattering length densities (contrast) of simulated points
    model_filename_list    : list of model names

    output
    plot                   : points_<model_filename>.png
    """

    ## figure settings
    markersize = 0.5
    max_l = get_max_dimension(x_list, y_list, z_list)*1.1
    lim = [-max_l, max_l]

    for x,y,z,p,model_filename,color in zip(x_list,y_list,z_list,sld_list,model_filename_list,colors):

        ## find indices of positive, zero and negatative contrast
        idx_neg = np.where(p < 0.0)
        idx_pos = np.where(p > 0.0)
        idx_nul = np.where(p == 0.0)

        f,ax = plt.subplots(1,3,figsize=(12,4))

        ## plot, perspective 1
        ax[0].plot(x[idx_pos], z[idx_pos], linestyle='none', marker='.', markersize=markersize, color=color)
        ax[0].plot(x[idx_neg], z[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[0].plot(x[idx_nul], z[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')
        ax[0].set_xlim(lim)
        ax[0].set_ylim(lim)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('z')
        ax[0].set_title('pointmodel, (x,z), "front"')

        ## plot, perspective 2
        ax[1].plot(y[idx_pos], z[idx_pos], linestyle='none', marker='.', markersize=markersize, color=color) 
        ax[1].plot(y[idx_neg], z[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[1].plot(y[idx_nul], z[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')
        ax[1].set_xlim(lim)
        ax[1].set_ylim(lim)
        ax[1].set_xlabel('y')
        ax[1].set_ylabel('z')
        ax[1].set_title('pointmodel, (y,z), "side"')

        ## plot, perspective 3
        ax[2].plot(x[idx_pos], y[idx_pos], linestyle='none', marker='.', markersize=markersize, color=color) 
        ax[2].plot(x[idx_neg], y[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[2].plot(x[idx_nul], y[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')    
        ax[2].set_xlim(lim)
        ax[2].set_ylim(lim)
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[2].set_title('pointmodel, (x,y), "bottom"')
    
        plt.tight_layout()
        plt.savefig('%s/points_%s.%s' % (model_filename,model_filename,filetype))
        plt.close()

def plot_results(q, r_list, pr_list, I_list, Isim_list, sigma_list, S_list, name_list, xscale_lin, filetype, colors):
    """
    plot results for all models:
    - p(r) 
    - calculated formfactor P(r) times structure factor S(q) on log-log or log-lin scale
    - simulated data on log-log or log-lin scale
    """
    __, ax = plt.subplots(1,3,figsize=(12,4))

    zo = 1
    for (r, pr, I, Isim, sigma, S, model_name,color) in zip (r_list, pr_list, I_list, Isim_list, sigma_list, S_list, name_list, colors):
        ax[0].plot(r,pr,zorder=zo, color=color, label='p(r), %s' % model_name)

        ax[2].errorbar(q,Isim,yerr=sigma,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % model_name,zorder=zo)

        if S[0] != 1.0 or S[-1] != 1.0:
            ax[1].plot(q, S, linestyle='--', color=color, zorder=0, label=r'$S(q)$, %s' % model_name)
            ax[1].plot(q, I,color=color, zorder=zo, label=r'$I(q)=P(q)S(q)$, %s' % model_name)
            ax[1].set_ylabel(r'$I(q)=P(q)S(q)$')
        else:
            ax[1].plot(q, I, zorder=zo, color=color, label=r'$P(q)=I(q)/I(0)$, %s' % model_name)
            ax[1].set_ylabel(r'$P(q)=I(q)/I(0)$')
        zo += 1

    ## figure settings, p(r)
    ax[0].set_xlabel(r'$r$ [$\mathrm{\AA}$]')
    ax[0].set_ylabel(r'$p(r)$')
    ax[0].set_title('pair distance distribution function')
    ax[0].legend(frameon=False)

    ## figure settings, calculated scattering
    if not xscale_lin:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[1].set_title('normalized scattering, no noise')
    ax[1].legend(frameon=False)

    ## figure settings, simulated scattering
    if not xscale_lin:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[2].set_ylabel(r'$I(q)$ [a.u.]')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot.' + filetype)
    plt.close()

def plot_fit(q, I_list, I_exp, sigma_exp, name_list, data_filename, xscale_lin, filetype, colors):
    """
    plot results for all models:
    - p(r) 
    - calculated formfactor P(r) times structure factor S(q) on log-log or log-lin scale
    - simulated data on log-log or log-lin scale
    """
    N_models = len(I_list)
    width = N_models*4
    __, ax = plt.subplots(1,N_models,figsize=(width,4))

    # estimate offset and scaling
    background = np.mean(I_exp[-5:])
    I0 = np.mean(I_exp[0:3])

    i = 0
    for (I, model_name,color) in zip (I_list, name_list, colors):
        if N_models == 1:
            p = ax
        else:
            p = ax[i]
        p.errorbar(q,I_exp,yerr=sigma_exp,linestyle='none',marker='.', color='grey',zorder=0, label = data_filename) #label=r'$I_\mathrm{exp}(q)$')
        I_model = I0 * I + background
        p.plot(q, I_model, zorder=1, color=color, label=r'%s' % model_name)
        p.set_ylabel(r'$I(q)$')
        i += 1

        if not xscale_lin:
            p.set_xscale('log')
        p.set_yscale('log')
        p.set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
        p.set_ylabel(r'$I(q)$ [a.u.]')
        # p.set_title('Data and fit')
        p.legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('fit.' + filetype)
    plt.close()
    
def plot_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list, filetype, colors):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    zo = 1
    for (d, G, Gsim, sigmaG, model_name, color) in zip (delta_list, G_list, Gsim_list, sigma_G_list, name_list, colors):

        ax[0].plot(d, G, zorder=zo, color=color,label=r'$G$, %s' % model_name)
        ax[0].set_ylabel(r'$G(\delta)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[0].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[0].set_title('theoretical SESANS, no noise')
        ax[0].legend(frameon=False)
                         

        ax[1].errorbar(d,Gsim,yerr=sigmaG,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % model_name,zorder=zo)
        ax[1].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[1].set_ylabel(r'$\ln(P)/(t\lambda^2)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[1].set_title('simulated SESANS, with noise')
        ax[1].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('sesans.' + filetype)
    plt.close()

def save_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list):

    for (d, G, Gsim, sigmaG, model_name) in zip (delta_list, G_list, Gsim_list, sigma_G_list, name_list):
         
        with open('%s/G_%s.ses' % (model_name,model_name),'w') as f:
            f.write('# Theoretical SESANS data\n')
            f.write('# %-12s %-12s\n' % ('delta','G'))
            for i in range(len(d)):
                f.write('  %-12.5e %-12.5e\n' % (d[i], G[i]))
        
        with open('%s/Gsim_%s.ses' % (model_name,model_name),'w') as f:
            f.write('# Simulated SESANS data, with noise\n')
            f.write('# %-12s %-12s %-12s\n' % ('delta','G','sigma_G'))
            for i in range(len(d)):
                f.write('  %-12.5e %-12.5e %-12.5e\n' % (d[i], Gsim[i], sigmaG[i]))

def generate_pdb(x_list, y_list, z_list, sld_list, model_filename_list):
    """
    Generates a visualisation file in PDB format with the simulated points (coordinates) and contrasts
    ONLY FOR VISUALIZATION!
    Each bead is represented as a dummy atom
    Carbon, C : positive contrast
    Hydrogen, H : zero contrast
    Oxygen, O : negateive contrast
    information of accurate contrasts not included, only sign
    IMPORTANT: IT WILL NOT GIVE THE CORRECT RESULTS IF SCATTERING IS CACLLUATED FROM THIS MODEL WITH E.G. CRYSOL, PEPSI-SAXS, FOXS, CAPP OR THE LIKE!
    """

    for (x,y,z,p,model_filename) in zip(x_list, y_list, z_list, sld_list, model_filename_list):
        with open('%s/%s.pdb' % (model_filename,model_filename),'w') as f:
            f.write('TITLE    POINT SCATTER FOR MODEL: %s\n' % model_filename)
            f.write('REMARK   GENERATED WITH Shape2SAS\n')
            f.write('REMARK   EACH BEAD REPRESENTED BY DUMMY ATOM\n')
            f.write('REMARK   CARBON, C : POSITIVE EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   HYDROGEN, H : ZERO EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   OXYGEN, O : NEGATIVE EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   ACCURATE SCATTERING LENGTH DENSITY INFORMATION NOT INCLUDED\n')
            f.write('REMARK   OBS: WILL NOT GIVE CORRECT RESULTS IF SCATTERING IS CALCULATED FROM THIS MODEL WITH E.G CRYSOL, PEPSI-SAXS, FOXS, CAPP OR THE LIKE!\n')
            f.write('REMARK   ONLY FOR VISUALIZATION, E.G. WITH PYMOL\n')
            f.write('REMARK    \n')
            for i in range(len(x)):
                if p[i] > 0:
                    atom = 'C'
                elif p[i] == 0:
                    atom = 'H'
                else:
                    atom = 'O'
                f.write('ATOM  %6i %s   ALA A%6i  %8.3f%8.3f%8.3f  1.00  0.00           %s \n'  % (i,atom,i,x[i],y[i],z[i],atom))
            f.write('END')

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

def str2bool(v):
    """
    Function to circumvent the argparse default behaviour 
    of not taking False inputs, when default=True.
    """
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def separate_string(arg):
    arg = re.split('[ ,]+', arg)
    return [str(i) for i in arg]

def float_list(arg):
    """
    Function to convert a string to a list of floats.
    Note that this function can interpret numbers with scientific notation 
    and negative numbers.

    input:
        arg: string, input string

    output:
        list of floats
    """
    arg = re.sub(r'\s+', ' ', arg.strip())
    arg = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", arg)
    return [float(i) for i in arg]

def check_3Dinput(input: list, default: list, name: str, N_subunits: int, i: int):
    """
    Function to check if 3D vector input matches 
    in lenght with the number of subunits

    input:
        input: list of floats, input values
        default: list of floats, default values

    output:
        list of floats
    """
    try:
        inputted = input[i]
        if len(inputted) != N_subunits:
            warnings.warn(f"The number of subunits and {name} do not match. Using {default}")
            inputted = default * N_subunits
    except:
        inputted = default * N_subunits
        #warnings.warn(f"Could not find {name}. Using default {default}.")

    return inputted

def check_input(input: float, default: float, name: str, i: int):
    """
    Function to check if input is given, 
    if not, use default value.

    input:
        input: float, input value
        default: float, default value
        name: string, name of the input

    output:
        float
    """
    try:
        inputted = input[i]
    except:
        inputted = default
        #warnings.warn(f"Could not find {name}. Using default {default}.")

    return inputted

@dataclass
class ModelPointDistribution:
    """
    Point distribution of a model
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sld: np.ndarray #scattering length density for each point
    volume_total: float

def getPointDistribution(subunit_type,sld,dimensions,com,rotation,exclude_overlap,Npoints):
    x_new, y_new, z_new, sld_new, volume_total = GenerateAllPoints(Npoints, com, subunit_type, dimensions, rotation, sld, exclude_overlap).onGeneratingAllPointsSeparately()
    return ModelPointDistribution(x=x_new, y=y_new, z=z_new, sld=sld_new, volume_total=volume_total)
