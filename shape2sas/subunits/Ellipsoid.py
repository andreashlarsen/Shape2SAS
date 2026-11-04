import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class Ellipsoid(Subunit):
    aliases = ["ellipsoid","ellips"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.c = dimensions

    def getVolume(self):
        """Returns the volume of an ellipsoid"""
        return (4 / 3) * np.pi * self.a * self.b * self.c

    def getHalfWidths(self):
        """Returns the half-widths of the box around an ellipsoid"""
        return self.a, self.b, self.c

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """check for points within a ellipsoid"""

        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2 + z_eff**2 / self.c**2
        idx = np.where(d2 > 1)

        return idx
