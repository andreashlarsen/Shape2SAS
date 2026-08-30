import numpy as np 
from .subunits_helpfunctions import check_dimension

class Cube:
    aliases = ["cube","dice"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,1)     
        self.a = dimensions[0]

    def getVolume(self):
        """Returns the volume of a cube"""
        return self.a**3

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cube"""
        # a is the full side length, so the generating box IS the cube:
        # no rejection needed, and the box volume matches getVolume()
        N = Npoints
        x_add = np.random.uniform(-self.a / 2, self.a / 2, N)
        y_add = np.random.uniform(-self.a / 2, self.a / 2, N)
        z_add = np.random.uniform(-self.a / 2, self.a / 2, N)

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cube"""
        idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2))
        
        return idx
