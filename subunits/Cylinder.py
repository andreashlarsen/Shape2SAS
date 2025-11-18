import numpy as np 
from .subunits_helpfunctions import check_dimension

class Cylinder:
    aliases = ["cylinder","disc","rod","cyl"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)     
        self.R,self.l = dimensions

    def getVolume(self):
        """Returns the volume of a cylinder"""

        return np.pi * self.R**2 * self.l
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cylinder"""

        Volume = self.getVolume()
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where(d < self.R)
        x_add,y_add,z_add = x[idx],y[idx],z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cylinder"""

        d = np.sqrt(x_eff**2+y_eff**2)
        idx = np.where((d > self.R) | (abs(z_eff) > self.l / 2))
        return idx

