import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class CylinderRing(Subunit):
    aliases = ["cylinderring","ring","cylring","discring","hollowcylinder","hollowdisc","hollowrod"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,3)     
        R,r,self.l = dimensions
        self.R,self.r = max(R, r), min(R, r)
    
    def getVolume(self):
        """Returns the volume of a cylinder ring"""
        return np.pi * (self.R**2 - self.r**2) * self.l

    def getHalfWidths(self):
        """Returns the half-widths of the box around a cylinder ring"""
        return self.R, self.R, self.l / 2

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cylinder ring"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        idx = np.where((d > self.R) | (d < self.r) | (abs(z_eff) > self.l / 2))
        return idx
