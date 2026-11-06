import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class Torus(Subunit):
    aliases = ["torus","toroid","doughnut"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,2)     
        self.R,self.r = dimensions

    def getVolume(self):
        """Returns the volume of a torus"""

        return 2 * np.pi**2 * self.r**2 * self.R

    def getHalfWidths(self):
        """Returns the half-widths of the box around a torus"""
        return self.R + self.r, self.R + self.r, self.r

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a torus"""
        # equation: (R-sqrt(x**2+y**2))**2 + z**2 = (R-d)**2 + z**2 = r
        d = np.sqrt(x_eff**2 + y_eff**2)
        idx = np.where((self.R-d)**2 + z_eff**2 > self.r**2)
        return idx
