import numpy as np
from .subunits_helpfunctions import Subunit, check_dimension

class Cylinder(Subunit):
    aliases = ["cylinder","rod","cyl"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,2)
        self.R,self.l = dimensions

    def getVolume(self):
        """Returns the volume of a cylinder"""

        return np.pi * self.R**2 * self.l

    def getHalfWidths(self):
        """Returns the half-widths of the box around a cylinder"""
        return self.R, self.R, self.l / 2

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cylinder"""

        d = np.sqrt(x_eff**2+y_eff**2)
        idx = np.where((d > self.R) | (abs(z_eff) > self.l / 2))
        return idx
