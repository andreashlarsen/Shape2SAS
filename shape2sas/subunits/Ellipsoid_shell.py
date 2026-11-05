import numpy as np 
from .subunits_helpfunctions import Subunit, check_dimension

class Ellipsoid_shell(Subunit):
    aliases = ["ellipsoid_shell","ellips_shell"]

    def __init__(self, dimensions, **properties):
        super().__init__(**properties)
        check_dimension(self.aliases[0],dimensions,4)     
        self.a,self.b,self.c,self.T = dimensions
        # define outer dimensions ( = semiaxis + half thickness):
        self.a_plus,self.b_plus,self.c_plus = self.a+self.T/2,self.b+self.T/2,self.c+self.T/2

    def getVolume(self):
        return (4 / 3) * np.pi * (self.a_plus * self.b_plus * self.c_plus - (self.a_plus-self.T) * (self.b_plus-self.T) * (self.c_plus-self.T))

    def getHalfWidths(self):
        """Returns the half-widths of the box around an ellipsoid shell"""
        return self.a_plus, self.b_plus, self.c_plus

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within an ellipsoid shell"""
        d2_max = x_eff**2 / self.a_plus**2 + y_eff**2 / self.b_plus**2 + z_eff**2 / self.c_plus**2
        d2_min = x_eff**2 / (self.a_plus-self.T)**2 + y_eff**2 / (self.b_plus-self.T)**2 + z_eff**2 / (self.c_plus-self.T)**2
        idx = np.where((d2_max > 1) | (d2_min < 1))

        return idx
