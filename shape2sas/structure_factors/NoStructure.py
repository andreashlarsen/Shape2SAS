import numpy as np 
from .StructureFactor import StructureFactor

class NoStructure(StructureFactor):
    """No structure factor, i.e. S(q) = 1: dilute, non-interacting particles"""
    aliases = ["none","no","nostructure","unity"]

    def structure_factor(self, q):
        """Returns the structure factor of a dilute solution"""
        return np.ones_like(q)

    def effective_structure_factor(self, q, point_distribution, Pq):
        """the particles do not interact, whatever their shape"""
        return np.ones_like(q)
