import numpy as np 
from .subunits_helpfunctions import check_dimension

from scipy.special import gamma

class Superellipsoid:
    aliases = ["superellipsoid","superellips"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,4)     
        self.R,self.eps,self.t,self.s = dimensions

    @staticmethod
    def beta(a, b):
        """beta function"""
        return gamma(a) * gamma(b) / gamma(a + b)

    def getVolume(self):
        """Returns the volume of a superellipsoid"""
        return (8 / (3 * self.t * self.s) * self.R**3 * self.eps * 
                self.beta(1 / self.s, 1 / self.s) * self.beta(2 / self.t, 1 / self.t))
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a superellipsoid"""
        Volume = self.getVolume()
        Volume_max = 2 * self.R * self.eps * 2 * self.R * 2 * self.R
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R * self.eps, self.R * self.eps, N)
        d = ((np.abs(x)**self.s + np.abs(y)**self.s)**(self.t/ self.s) 
            + np.abs(z / self.eps)**self.t)
        idx = np.where(d < np.abs(self.R)**self.t)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within the subunit"""
        d = ((np.abs(x_eff)**self.s + np.abs(y_eff)**self.s)**(self.t / self.s) 
        + np.abs(z_eff / self.eps)**self.t)
        idx = np.where(d >= np.abs(self.R)**self.t)
        return idx
