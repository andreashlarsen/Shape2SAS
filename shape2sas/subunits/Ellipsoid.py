import numpy as np 
from .Subunit import Subunit

class Ellipsoid(Subunit):
    aliases = ["ellipsoid","ellips"]

    def __init__(self, a, b, c):
        """semi-axes along x, y and z"""
        self.a,self.b,self.c = a,b,c

    def volume(self):
        """Returns the volume of an ellipsoid"""
        return (4 / 3) * np.pi * self.a * self.b * self.c

    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in an ellipsoid"""
        Volume = self.volume()
        Volume_max = 2 * self.a * 2 * self.b * 2 * self.c
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = rng.uniform(-self.a, self.a, N)
        y = rng.uniform(-self.b, self.b, N)
        z = rng.uniform(-self.c, self.c, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the ellipsoid"""

        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2 + z_eff**2 / self.c**2
        idx = (d2 > 1)

        return idx
