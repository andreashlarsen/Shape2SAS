import numpy as np
from .subunits_helpfunctions import Subunit, check_dimension

class HollowSphere(Subunit):
    aliases = ["hollowsphere","shell"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,2)
        self.R,self.r = max(dimensions), min(dimensions)

    def getVolume(self):
        """Returns the volume of a hollow sphere"""
        return (4 / 3) * np.pi * (self.R**3 - self.r**3)

    def getHalfWidths(self):
        """Returns the half-widths of the box around a hollow sphere"""
        return self.R, self.R, self.R

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a hollow sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        idx = np.where((d > self.R) | (d < self.r))
        return idx
