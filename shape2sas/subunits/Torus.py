import numpy as np 
from .Subunit import Subunit

class Torus(Subunit):
    aliases = ["torus","toroid","doughnut"]

    def __init__(self, radius, tube_radius):
        """the overall radius, to the centre of the tube, and the radius of the tube's cross section"""
        self.R,self.r = radius,tube_radius

    def volume(self):
        """Returns the volume of a torus"""

        return 2 * np.pi**2 * self.r**2 * self.R

    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a torus"""
        Volume = self.volume()
        L = 2 * (self.R + self.r)
        l = 2 * self.r
        Volume_max = L*L*l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = rng.uniform(-L/2, L/2, N)
        y = rng.uniform(-L/2, L/2, N)
        z = rng.uniform(-l/2, l/ 2, N)
        # equation: (R-sqrt(x**2+y**2))**2 + z**2 = (R-d)**2 + z**2 = r
        d = np.sqrt(x**2 + y**2)
        idx = np.where((self.R-d)**2 + z**2 < self.r**2)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the torus"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        idx = ((self.R-d)**2 + z_eff**2 > self.r**2)
        return idx
