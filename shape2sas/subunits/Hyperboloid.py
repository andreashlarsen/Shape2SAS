import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class Hyperboloid(Subunit):
    aliases = ["hyperboloid", "hourglass", "coolingtower"]
    
    # https://mathworld.wolfram.com/One-SheetedHyperboloid.html
    # https://www.vcalc.com/wiki/hyperboloid-volume

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,3)     
        self.r,self.c,self.h = dimensions

    def getVolume(self):
        """Returns the volume of a hyperboloid"""
        return np.pi * 2*self.h * self.r**2 * ( 1 + (2*self.h)**2 / ( 12 * self.c**2 ) )

    def getHalfWidths(self):
        """Returns the half-widths of the box around a hyperboloid"""
        R = self.r * np.sqrt( 1 + self.h**2 / self.c**2 ) # radius at the ends, z = +-h
        return R, R, self.h

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Hyperboloid"""
        idx = np.where((x_eff**2/self.r**2 + y_eff**2/self.r**2 - z_eff**2/self.c**2 > 1.0) | (abs(z_eff) > self.h))
        return idx
