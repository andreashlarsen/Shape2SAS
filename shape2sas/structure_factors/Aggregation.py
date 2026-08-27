import numpy as np 
from .structure_factors_helpfunctions import check_Spar, decoupling_approx

class Aggregation:
    """Fractal aggregate structure factor with dimensionality 2"""
    aliases = ["aggregation","aggr","aggregate","frac2d"]
    par_names = ["R_eff","N_aggr","fraction_aggregated"]

    def __init__(self, S_par):
        check_Spar(self.aliases[0],S_par,self.par_names)
        self.Reff,self.Naggr,self.fracs_aggr = S_par

    def calc_S(self, q):
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
        qR = q * self.Reff
        S_aggr = 1 + (self.Naggr - 1)/(1 + qR**2 * self.Naggr / 3)
        return S_aggr

    def structure_eff(self, q, point_distribution, Pq):
        """Return effective structure factor for aggregation"""
        S = self.calc_S(q)
        S_eff = decoupling_approx(q,point_distribution,Pq,S)
        S_eff = (1 - self.fracs_aggr) + self.fracs_aggr * S_eff
        return S_eff

    @staticmethod
    def getSesansRange(S_par, dmax):
        """the aggregate, not the single particle, sets the length scale"""
        Reff = S_par[0]
        qmin = 0.001 * np.pi / (2 * Reff)
        deltamax = 3 * Reff
        return qmin, deltamax
