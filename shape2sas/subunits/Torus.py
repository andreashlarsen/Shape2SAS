import numpy as np 
from .subunits_helpfunctions import check_dimension

class Torus:
    aliases = ["torus","toroid","doughnut"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)     
        self.R,self.r = dimensions

    def getVolume(self):
        """Returns the volume of a torus"""

        return 2 * np.pi**2 * self.r**2 * self.R

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a torus"""
        Volume = self.getVolume()
        L = 2 * (self.R + self.r)
        l = 2 * self.r
        Volume_max = L*L*l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-L/2, L/2, N)
        y = np.random.uniform(-L/2, L/2, N)
        z = np.random.uniform(-l/2, l/ 2, N)
        # equation: (R-sqrt(x**2+y**2))**2 + z**2 = (R-d)**2 + z**2 = r
        d = np.sqrt(x**2 + y**2)
        idx = np.where((self.R-d)**2 + z**2 < self.r**2)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a torus"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        idx = np.where((self.R-d)**2 + z_eff**2 > self.r**2)
        return idx
