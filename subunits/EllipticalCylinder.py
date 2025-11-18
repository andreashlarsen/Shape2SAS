import numpy as np 
from .subunits_helpfunctions import check_dimension

class EllipticalCylinder:
    aliases = ["ellipticalcylinder","ellipticalcyl","ellipticalrod"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.l = dimensions

    def getVolume(self):
        """Returns the volume of an elliptical cylinder"""
        return np.pi * self.a * self.b * self.l

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of an elliptical cylinder"""

        Volume = self.getVolume()
        Volume_max = 2 * self.a * 2 * self.b * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a, self.a, N)
        y = np.random.uniform(-self.b, self.b, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add 

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Elliptical cylinder"""
        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2
        idx = np.where((d2 > 1) | (abs(z_eff) > self.l / 2))
        return idx
