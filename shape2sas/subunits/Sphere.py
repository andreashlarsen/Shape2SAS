import numpy as np
from .subunits_helpfunctions import Subunit, check_dimension

class Sphere(Subunit):
    aliases = ["sphere","ball","sph"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,1)
        self.R = dimensions[0]

    def getVolume(self) -> float:
        """Returns the volume of a sphere"""
        return (4 / 3) * np.pi * self.R**3

    def getHalfWidths(self):
        """Returns the half-widths of the box around a sphere"""
        return self.R, self.R, self.R

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        idx = np.where(d > self.R)
        return idx
