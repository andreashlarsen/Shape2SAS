import numpy as np 
from .structure_factors_helpfunctions import check_Spar, decoupling_approx, default_sesans_range

class HardSphere:
    """Hard-sphere structure factor, Percus-Yevick approximation"""
    aliases = ["hardsphere","hs","hard-sphere"]
    par_names = ["concentration","R_HS"]

    def __init__(self, S_par):
        check_Spar(self.aliases[0],S_par,self.par_names)
        self.conc,self.R_HS = S_par

    def calc_S(self, q):
        """
        Calculate the hard-sphere structure factor using the Percus-Yevick approximation.
        Implements the stable version with Taylor expansion for small A = 2*R*q.
        adapted directly from the sasview code
        """

        if self.conc <= 0.0:
            return np.ones(len(q))

        vf = self.conc
        R = self.R_HS
        X = np.abs(2.0 * R * q)  # X = 2*R*q

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

    def structure_eff(self, q, point_distribution, Pq):
        """Return effective structure factor for hard spheres"""
        S = self.calc_S(q)
        return decoupling_approx(q,point_distribution,Pq,S)

    @staticmethod
    def getSesansRange(S_par, dmax):
        return default_sesans_range(dmax)
