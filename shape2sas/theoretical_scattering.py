"""Theoretical scattering: pair distances, p(r), the form factor P(q)
and the resulting intensity I(q)."""

import logging
import os
from dataclasses import dataclass

import numpy as np
from fast_histogram import histogram1d #histogram1d from fast_histogram is faster than np.histogram (https://pypi.org/project/fast-histogram/) 

from .helpfunctions import getStructureFactorClass, sinc

logger = logging.getLogger(__name__)

class PairDistanceDistribution:
    """
    pair distance distribution function, p(r), of a point distribution, and the form factor, P(q), and forward scattering, P(0), derived from it

    the scattering amplitude is the integral of the excess scattering length density over the particle volume, V,
        A(q) = int_V Delta rho(r) exp(i q.r) dr
    which is estimated by Monte Carlo integration over the N points sampled uniformly in V, with contrasts w_i:
        A(q) ~ (V/N) sum_i w_i exp(i q.r_i)
    V/N is the volume element of this estimate; the points are integration nodes, not scatterers with a form factor of their own.
    the orientational average of |A(q)|^2 is a Debye sum over pairs of points, calculated from the contrast-weighted histogram of pair distances, h(r).
    the squared sum contains self-terms (i = j), which vanish as 1/N for N -> infinity; leaving them out speeds up convergence,
    so h(r) is summed over the unordered pairs i < j only.

    h(r) is kept at full resolution, since P(q) is most accurate when calculated from it.
    the AUSAXS backend calculates h(r) on a fine grid, which is undersampled for presentation, so p(r) itself has neighbouring bins merged to roughly prpoints bins.

    input:
    point_distribution : points and their scattering length densities
    prpoints           : number of bins in p(r)
    polydispersity     : relative polydispersity, float
    use_ausaxs         : try the AUSAXS backend first, and fall back to the default method if pyausaxs is not installed or fails

    attributes:
    r, pr      : p(r), with roughly prpoints bins, normalized so pr_max = 1
    r_hist, hr : full-resolution h(r), summed over unordered pairs i < j (no self-terms)
    dmax       : maximum distance
    Rg         : radius of gyration
    """

    @staticmethod
    def calc_all_dist(point_distribution):
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

    @staticmethod
    def calc_all_contrasts(point_distribution):
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

    @staticmethod
    def generate_histogram(dist, prpoints, contrast, r_max):
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

    @staticmethod
    def merge_bins(r, h, r_max, prpoints):
        """
        merge every N neighbouring bins of an equidistant histogram, h(r), so the merged histogram has roughly prpoints bins up to r_max
        the counts are summed rather than rebinned onto a new grid, since rebinning gives severe aliasing errors
        a histogram which already has prpoints bins, or which has fewer than two bins (e.g. zero contrast), is returned as-is

        input
        r        : equidistant bin distances
        h        : histogram
        r_max    : max distance included in the histogram
        prpoints : requested number of bins

        output
        r        : distances of merged bins
        h        : merged histogram
        """
        if len(r) < 2:
            return r, h

        bin_width = r[1] - r[0]
        n_merge = max(1, int(round(r_max / (bin_width * prpoints))))
        if n_merge == 1:
            return r, h

        # pad with empty bins so the length is a multiple of n_merge
        n_padded = -(-len(h) // n_merge) * n_merge
        h = np.concatenate((h, np.zeros(n_padded - len(h), dtype=h.dtype)))
        r = r[0] + bin_width * np.arange(n_padded, dtype=r.dtype)

        return r.reshape(-1, n_merge).mean(axis=1), h.reshape(-1, n_merge).sum(axis=1)

    def calc_hr(self, dist, contrast, prebinned=False):
        """
        calculate h(r)
        h(r) is the contrast-weighted histogram of distances

        input:
        dist      : pair distances
        contrast  : contrast weight of each distance
        prebinned : dist/contrast are already a histogram, on a grid finer than prpoints

        output:
        r         : distances of bins
        hr        : pair distance distribution function
        dmax      : maximum distance
        r_max     : max distance included in the histogram
        """
        prpoints, polydispersity = self.prpoints, self.polydispersity
        if dist.dtype != np.float32:
            dist = dist.astype(np.float32, copy=False)
        if contrast.dtype != np.float32:
            contrast = contrast.astype(np.float32, copy=False)

        ## make r range in h(r) histogram slightly larger than dmax
        ratio_rmax_dmax = 1.05

        ## determine dmax, the maximum distance to include in the histogram
        lognormal = False
        if polydispersity > 0.0:
            if lognormal:
                dmax = np.amax(dist)*np.exp(3* polydispersity)
            else:
                dmax = np.amax(dist) * (1 + 3 * polydispersity)
        else:
            dmax = np.amax(dist)
        r_max = dmax * ratio_rmax_dmax

        ## calc h(r) with/without polydispersity
        if polydispersity > 0.0:
            N_poly_integral = 25 # should be uneven to include 1 in factor_range (precalculated)
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
            if prebinned:
                bin_width = dist[1] - dist[0]
                n_extra = int(np.ceil(r_max / bin_width)) + 1 - len(dist)
                r = np.concatenate((dist, dist[-1] + bin_width * np.arange(1, n_extra + 1, dtype=np.float32)))
                nbins = len(r)
                hr = np.zeros(nbins, dtype=np.float32)
                # with the prebinned histogram, we can just shift the bins by the scale factor
                for i, factor_d in enumerate(factor_range):
                    bin_index = np.rint(dist * factor_d / bin_width).astype(np.int64)
                    hr += np.bincount(bin_index, weights=contrast, minlength=nbins)[:nbins] * norm_range[i]
            else:
                r, hr_1 = self.generate_histogram(dist, prpoints, contrast, r_max)
                hr  = np.zeros_like(hr_1, dtype=np.float32)
                #norm = 0.0
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
        elif prebinned: # dist and contrast are already binned into a fine histogram, so just return them as-is
            r, hr = dist, contrast
        else:
            r, hr = self.generate_histogram(dist, prpoints, contrast, r_max)

        return r, hr, dmax, r_max

    @staticmethod
    def calc_Rg(r, pr):
        """ 
        calculate Rg from r and p(r)
        """
        sum_pr_r2 = np.sum(pr * r**2)
        sum_pr = np.sum(pr)
        Rg = np.sqrt(abs(sum_pr_r2 / sum_pr) / 2)

        return Rg

    def calc_pair_distances(self, use_ausaxs=True):
        """
        calculate the contrast-weighted pair distances of a point distribution

        input:
        use_ausaxs         : try the AUSAXS backend first, and fall back to the
                             default method if pyausaxs is not installed or fails

        output:
        dist      : pair distances
        contrast  : contrast weight of each distance
        prebinned : whether dist/contrast are already a histogram (AUSAXS) rather than
                    individual pairs (default)
        """
        point_distribution = self.point_distribution
        if use_ausaxs:
            try:
                from .ausaxs_debye import distance_histogram
                logger.info('        calculating distance histogram (AUSAXS backend)...')
                r, hr = distance_histogram(
                    np.concatenate(point_distribution.x),
                    np.concatenate(point_distribution.y),
                    np.concatenate(point_distribution.z),
                    np.concatenate(point_distribution.sld)
                )
                return r, hr, True
            except ImportError as e:
                logger.info(f"        AUSAXS backend not installed ({e}); using default distance calculation.")
            except Exception as e:
                logger.warning(f"        AUSAXS backend failed ({type(e).__name__}: {e}); using default distance calculation.")

        logger.info('        calculating distances...')
        dist = self.calc_all_dist(point_distribution)
        logger.info('        calculating contrasts...')
        contrast = self.calc_all_contrasts(point_distribution)

        return dist, contrast, False

    def __init__(self, point_distribution, prpoints=100, polydispersity=0.0, use_ausaxs=True):
        self.point_distribution = point_distribution
        self.prpoints = prpoints
        self.polydispersity = polydispersity

        dist, contrast, prebinned = self.calc_pair_distances(use_ausaxs)

        logger.info('        calculating p(r)...')
        self.r_hist, self.hr, self.dmax, r_max = self.calc_hr(dist, contrast, prebinned)
        logger.info(f"           dmax: {self.dmax:.3e} A")

        r, pr = self.merge_bins(self.r_hist, self.hr, r_max, prpoints)
        self.r, self.pr = r, pr / np.amax(pr)

        self.Rg = self.calc_Rg(self.r_hist, self.hr)
        logger.info(f"           Rg  : {self.Rg:.3e} A")

    def calc_Pq(self, q):
        """
        calculate form factor, P(q), normalized so P(0) = 1

        input:
        q    : scattering vector

        output:
        Pq   : form factor, normalized so P(0) = 1
        """
        ## calculate P(q) from the full-resolution h(r)
        Pq = 0
        for (r_i, hr_i) in zip(self.r_hist, self.hr):
            qr = q * r_i
            Pq += hr_i * sinc(qr)

        # normalization, P(0) = 1
        sum_hr = np.sum(self.hr, dtype=np.float64)
        if sum_hr == 0:
            Pq /= 1E-5
        else:
            Pq /= abs(sum_hr)

        return Pq

    def calc_P0(self):
        """
        calculate forward scattering per unit volume fraction, P(0), so that I(q) = conc * P(0) * P(q) * S(q)

        P(0) = |A(0)|^2 / V = V * <w>^2 (see the class docstring), where the squared mean contrast is estimated from h(r), i.e. without the self-terms:
            <w>^2 = 2 sum(h) / (N (N - 1))
        which is independent of the binning of h(r).
        with contrast in units of 1e-6 A^-2, the factor 1E-4 converts A^-1 to cm^-1

        output:
        P0   : forward scattering per unit volume fraction [cm^-1]
        """
        ## the estimate of <w>^2 can come out slightly negative when the net contrast is close to zero
        sum_hr = np.sum(self.hr, dtype=np.float64)
        N = len(np.concatenate(self.point_distribution.x))
        contrast2 = max(0.0, 2 * sum_hr / (N * (N - 1)))
        return self.point_distribution.volume_total * contrast2 * 1E-4

    def save(self, model_filename):
        """save p(r)"""
        save_pr_func(self.r, self.pr, model_filename)

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

def uniform_q(qmin=0.001, qmax=0.5, qpoints=400):
    """equidistant q values from qmin to qmax [1/A]"""
    return np.linspace(qmin, qmax, qpoints)

@dataclass
class TheoreticalScattering:
    """
    the scattering of a model system, I(q) = conc * Pq * S_eff * exp(-(q sigma_r)^2 / 2)

    q     : q values [1/A]
    I     : intensity [cm^-1]
    Pq    : form factor per unit volume fraction [cm^-1], so Pq[q -> 0] = V * contrast^2
    S_eff : effective structure factor (decoupling approximation)
    """

    q: np.ndarray
    I: np.ndarray
    Pq: np.ndarray
    S_eff: np.ndarray

def get_theoretical_scattering(pdd, q, structure_factor="none", structure_factor_parameters=(), conc=0.02, sigma_r=0.0):
    """
    calculate the scattering, in absolute units, of particles with the pair distance distribution pdd

    pdd                         : PairDistanceDistribution of the particle
    q                           : q values [1/A]
    structure_factor            : "none", "hardsphere" or "aggregation" (or any of their aliases)
    structure_factor_parameters : parameters of the structure factor, see its par_names
                                  (hardsphere: concentration, R_HS; aggregation: R_eff, N_aggr, fraction_aggregated)
    conc                        : volume fraction of the particles
    sigma_r                     : interface roughness [A]

    raises ValueError on an unknown structure factor or a wrong number of its parameters
    """
    q = np.asarray(q, dtype=float)

    # the decoupling approximation needs the form factor normalized so P(0) = 1
    Pq_norm = pdd.calc_Pq(q)
    S_eff = calc_S_func(q, pdd.point_distribution, structure_factor, structure_factor_parameters, Pq_norm)
    P0 = pdd.calc_P0()
    I = conc * P0 * calc_Iq_func(q, Pq_norm, S_eff, sigma_r)

    return TheoreticalScattering(q=q, I=I, Pq=P0 * Pq_norm, S_eff=S_eff)

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

