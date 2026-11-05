import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

from scipy.special import gamma

class Superellipsoid(Subunit):
    aliases = ["superellipsoid","superellips"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
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

    def getHalfWidths(self):
        """Returns the half-widths of the box around a superellipsoid"""
        return abs(self.R), abs(self.R), abs(self.R * self.eps)
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within the subunit"""
        d = ((np.abs(x_eff)**self.s + np.abs(y_eff)**self.s)**(self.t / self.s) 
        + np.abs(z_eff / self.eps)**self.t)
        idx = np.where(d >= np.abs(self.R)**self.t)
        return idx
