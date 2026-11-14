import numpy as np 
from .Subunit import Subunit

from scipy.special import gamma

class Superellipsoid(Subunit):
    aliases = ["superellipsoid","superellips"]

    def __init__(self, radius, eccentricity, t, s):
        """the equatorial radius, the eccentricity (polar over equatorial radius), and the shape parameters t and s"""
        self.R,self.eps,self.t,self.s = radius,eccentricity,t,s

    @staticmethod
    def beta(a, b):
        """beta function"""
        return gamma(a) * gamma(b) / gamma(a + b)

    def volume(self):
        """Returns the volume of a superellipsoid"""
        return (8 / (3 * self.t * self.s) * self.R**3 * self.eps * 
                self.beta(1 / self.s, 1 / self.s) * self.beta(2 / self.t, 1 / self.t))
    
    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a superellipsoid"""
        Volume = self.volume()
        Volume_max = 2 * self.R * self.eps * 2 * self.R * 2 * self.R
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = rng.uniform(-self.R, self.R, N)
        y = rng.uniform(-self.R, self.R, N)
        z = rng.uniform(-self.R * self.eps, self.R * self.eps, N)
        d = ((np.abs(x)**self.s + np.abs(y)**self.s)**(self.t/ self.s) 
            + np.abs(z / self.eps)**self.t)
        idx = np.where(d < np.abs(self.R)**self.t)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add
    
    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the superellipsoid"""
        d = ((np.abs(x_eff)**self.s + np.abs(y_eff)**self.s)**(self.t / self.s) 
        + np.abs(z_eff / self.eps)**self.t)
        idx = (d >= np.abs(self.R)**self.t)
        return idx
