import numpy as np 
from .subunits_helpfunctions import check_dimension

class Cuboid:
    aliases = ["cuboid","brick"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.c = dimensions

    def getVolume(self):
        """Returns the volume of a cuboid"""
        return self.a * self.b * self.c
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cuboid"""
        Volume = self.getVolume()
        x_add = np.random.uniform(-self.a, self.a, Npoints)
        y_add = np.random.uniform(-self.b, self.b, Npoints)
        z_add = np.random.uniform(-self.c, self.c, Npoints)
        return x_add, y_add, z_add
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Cuboid"""
        idx = np.where((abs(x_eff) >= self.a/2) 
        | (abs(y_eff) >= self.b/2) | (abs(z_eff) >= self.c/2))
        return idx
