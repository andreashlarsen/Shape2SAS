import numpy as np 
from .Subunit import Subunit

class Sphere(Subunit):
    aliases = ["sphere","ball","sph"]

    def __init__(self, radius):
        self.R = radius

    def getVolume(self) -> float:
        """Returns the volume of a sphere"""
        return (4 / 3) * np.pi * self.R**3
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a sphere"""

        Volume = self.getVolume()
        Volume_max = (2*self.R)**3 # box around sphere.
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where(d < self.R) #save points inside sphere
        x_add,y_add,z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        idx = np.where(d > self.R)
        return idx
