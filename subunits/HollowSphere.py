import numpy as np 
from .subunits_helpfunctions import check_dimension

class HollowSphere:
    aliases = ["hollowsphere","shell"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)
        self.R,self.r = dimensions

    def getVolume(self):
        """Returns the volume of a hollow sphere"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 4 * np.pi * self.R**2 #surface area of a sphere
        else: 
            return (4 / 3) * np.pi * (self.R**3 - self.r**3)

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a hollow sphere"""
        Volume = self.getVolume()
        if self.r == self.R:
            #The hollow sphere is a shell if r=R
            phi = np.random.uniform(0,2 * np.pi, Npoints)
            costheta = np.random.uniform(-1, 1, Npoints)
            theta = np.arccos(costheta)

            x_add = self.R * np.sin(theta) * np.cos(phi)
            y_add = self.R * np.sin(theta) * np.sin(phi)
            z_add = self.R * np.cos(theta)
            return x_add, y_add, z_add
        Volume_max = (2*self.R)**3
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a hollow sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        if self.r > self.R:
             self.r, self.R = self.R, self.r
        if self.r == self.R:
            idx = np.where(d != self.R)
            return idx
        else:
            idx = np.where((d > self.R) | (d < self.r))
            return idx
