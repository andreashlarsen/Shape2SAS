import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class Cuboid(Subunit):
    aliases = ["cuboid","brick"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.c = dimensions

    def getVolume(self):
        """Returns the volume of a cuboid"""
        return self.a * self.b * self.c

    def getHalfWidths(self):
        """Returns the half-widths of the box around a cuboid"""
        return self.a / 2, self.b / 2, self.c / 2
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Cuboid"""
        idx = np.where((abs(x_eff) >= self.a/2) 
        | (abs(y_eff) >= self.b/2) | (abs(z_eff) >= self.c/2))
        return idx
