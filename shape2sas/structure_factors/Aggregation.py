import numpy as np 
from .StructureFactor import StructureFactor
from .structure_factors_helpfunctions import decoupling_approx

class Aggregation(StructureFactor):
    """Fractal aggregate structure factor with dimensionality 2"""
    aliases = ["aggregation","aggr","aggregate","frac2d"]

    def __init__(self, effective_radius, particles_per_aggregate, fraction_aggregated):
        """the effective radius of one particle [A], the number of particles per aggregate, and the fraction of particles in aggregates"""
        self.Reff,self.Naggr,self.fracs_aggr = effective_radius,particles_per_aggregate,fraction_aggregated

    def structure_factor(self, q):
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

    def effective_structure_factor(self, q, point_distribution, Pq):
        """only the aggregated fraction of the particles has structure"""
        S_eff = decoupling_approx(q,point_distribution,Pq,self.structure_factor(q))
        return (1 - self.fracs_aggr) + self.fracs_aggr * S_eff

    def sesans_range(self, dmax):
        """the aggregate, not the single particle, sets the length scale"""
        qmin = 0.001 * np.pi / (2 * self.Reff)
        deltamax = 3 * self.Reff
        return qmin, deltamax
