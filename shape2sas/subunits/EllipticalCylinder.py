import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class EllipticalCylinder(Subunit):
    aliases = ["ellipticalcylinder","ellipticalcyl","ellipticalrod"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.l = dimensions

    def getVolume(self):
        """Returns the volume of an elliptical cylinder"""
        return np.pi * self.a * self.b * self.l

    def getHalfWidths(self):
        """Returns the half-widths of the box around an elliptical cylinder"""
        return self.a, self.b, self.l / 2

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Elliptical cylinder"""
        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2
        idx = np.where((d2 > 1) | (abs(z_eff) > self.l / 2))
        return idx
