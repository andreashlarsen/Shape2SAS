import numpy as np 
from .subunits_helpfunctions import check_dimension

class Ellipsoid_shell:
    aliases = ["ellipsoid_shell","ellips_shell"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,4)     
        self.a,self.b,self.c,self.T = dimensions
        # define outer dimensions ( = semiaxis + half thickness):
        self.a_plus,self.b_plus,self.c_plus = self.a+self.T/2,self.b+self.T/2,self.c+self.T/2

    def getVolume(self):
        return (4 / 3) * np.pi * (self.a_plus * self.b_plus * self.c_plus - (self.a_plus-self.T) * (self.b_plus-self.T) * (self.c_plus-self.T))

    def getPointDistribution(self, Npoints):
        Volume = self.getVolume()
        Volume_max = 2 * self.a_plus * 2 * self.b_plus * 2 * self.c_plus
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a_plus, self.a_plus, N)
        y = np.random.uniform(-self.b_plus, self.b_plus, N)
        z = np.random.uniform(-self.c_plus, self.c_plus, N)

        d2_max = x**2 / self.a_plus**2 + y**2 / self.b_plus**2 + z**2 / self.c_plus**2
        d2_min = x**2 / (self.a_plus-self.T)**2 + y**2 / (self.b_plus-self.T)**2 + z**2 / (self.c_plus-self.T)**2
        idx = np.where((d2_max < 1) & (d2_min > 1))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within an ellipsoid shell"""
        d2_max = x_eff**2 / self.a_plus**2 + y_eff**2 / self.b_plus**2 + z_eff**2 / self.c_plus**2
        d2_min = x_eff**2 / (self.a_plus-self.T)**2 + y_eff**2 / (self.b_plus-self.T)**2 + z_eff**2 / (self.c_plus-self.T)**2
        idx = np.where((d2_max > 1) | (d2_min < 1))

        return idx
