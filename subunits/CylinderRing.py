import numpy as np 
from .subunits_helpfunctions import check_dimension

class CylinderRing:
    aliases = ["cylinderring","ring","cylring","discring","hollowcylinder","hollowdisc","hollowrod"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.R,self.r,self.l = dimensions
    
    def getVolume(self):
        """Returns the volume of a cylinder ring"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 2 * np.pi * self.R * self.l #surface area of a cylinder
        else: 
            return np.pi * (self.R**2 - self.r**2) * self.l

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cylinder ring"""
        Volume = self.getVolume()
        if self.r == self.R:
            #The cylinder ring is a shell
            phi = np.random.uniform(0, 2 * np.pi, Npoints)
            x_add = self.R * np.cos(phi)
            y_add = self.R * np.sin(phi)
            z_add = np.random.uniform(-self.l / 2, self.l / 2, Npoints)
            return x_add, y_add, z_add
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cylinder ring"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            idx = np.where((d != self.R) | (abs(z_eff) > self.l / 2))
            return idx
        else: 
            idx = np.where((d > self.R) | (d < self.r) | (abs(z_eff) > self.l / 2))
            return idx
