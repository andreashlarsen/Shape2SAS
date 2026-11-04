import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class HollowCube(Subunit):
    aliases = ["hollowcube","hollowdice"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,2)     
        self.a,self.b = max(dimensions), min(dimensions)

    def getVolume(self) :
        """Returns the volume of a hollow cube"""
        return self.a**3 - self.b**3

    def getHalfWidths(self):
        """Returns the half-widths of the box around a hollow cube"""
        return self.a / 2, self.a / 2, self.a / 2
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a hollow cube"""
        idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
        (abs(z_eff) >= self.a/2) | ((abs(x_eff) <= self.b/2) 
        & (abs(y_eff) <= self.b/2) & (abs(z_eff) <= self.b/2)))

        return idx
