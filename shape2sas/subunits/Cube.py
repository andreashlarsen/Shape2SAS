import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class Cube(Subunit):
    aliases = ["cube","dice"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,1)     
        self.a = dimensions[0]

    def getVolume(self):
        """Returns the volume of a cube"""
        return self.a**3

    def getHalfWidths(self):
        """Returns the half-widths of the box around a cube"""
        return self.a / 2, self.a / 2, self.a / 2

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cube"""
        idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2))
        
        return idx
