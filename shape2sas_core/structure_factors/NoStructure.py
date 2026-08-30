import numpy as np 
from shape2sas_core.sesans.sesans_helpfunctions import default_sesans_range

class NoStructure:
    """No structure factor, i.e. S(q) = 1: dilute, non-interacting particles"""
    aliases = ["none","no","nostructure","unity"]
    par_names = []

    def __init__(self, S_par=None):
        # takes no parameters - any given are ignored
        pass

    def calc_S(self, q):
        """Returns the structure factor of a dilute solution"""
        return np.ones_like(q)

    def structure_eff(self, q, point_distribution, Pq):
        """Return effective structure factor for no structure"""
        return np.ones_like(q)

    @staticmethod
    def getSesansRange(S_par, dmax):
        return default_sesans_range(dmax)
