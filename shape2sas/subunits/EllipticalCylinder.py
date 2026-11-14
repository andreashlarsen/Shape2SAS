import numpy as np 
from .Subunit import Subunit

class EllipticalCylinder(Subunit):
    aliases = ["ellipticalcylinder","ellipticalcyl","ellipticalrod"]

    def __init__(self, a, b, length):
        """semi-axes of the cross section along x and y, and the length along z"""
        self.a,self.b,self.l = a,b,length

    def volume(self):
        """Returns the volume of an elliptical cylinder"""
        return np.pi * self.a * self.b * self.l

    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in an elliptical cylinder"""

        Volume = self.volume()
        Volume_max = 2 * self.a * 2 * self.b * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = rng.uniform(-self.a, self.a, N)
        y = rng.uniform(-self.b, self.b, N)
        z = rng.uniform(-self.l / 2, self.l / 2, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add 

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the elliptical cylinder"""
        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2
        idx = ((d2 > 1) | (abs(z_eff) > self.l / 2))
        return idx
