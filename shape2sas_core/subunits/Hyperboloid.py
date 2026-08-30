import numpy as np 
from .subunits_helpfunctions import check_dimension

class Hyperboloid:
    aliases = ["hyperboloid", "hourglass", "coolingtower"]
    
    # https://mathworld.wolfram.com/One-SheetedHyperboloid.html
    # https://www.vcalc.com/wiki/hyperboloid-volume

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.r,self.c,self.h = dimensions

    def getVolume(self):
        """Returns the volume of a hyperboloid"""
        return np.pi * 2*self.h * self.r**2 * ( 1 + (2*self.h)**2 / ( 12 * self.c**2 ) )

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a hyperboloid"""
        L = 2 * self.h
        R = self.r * np.sqrt( 1 + L**2 / (4 * self.c**2 ) )
        Volume_max = 2*self.h * 2*R * 2*R
        Volume = np.pi * L * ( 2 * self.r**2 + R**2 ) / 3
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-R, R, N)
        y = np.random.uniform(-R, R, N)
        z = np.random.uniform(-self.h, self.h, N)
        idx = np.where(x**2/self.r**2 + y**2/self.r**2 - z**2/self.c**2 < 1.0)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Hyperboloid"""
        idx = np.where(x_eff**2/self.r**2 + y_eff**2/self.r**2 - z_eff**2/self.c**2 > 1.0)
        return idx
