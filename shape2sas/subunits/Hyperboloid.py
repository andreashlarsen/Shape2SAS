import numpy as np 
from .Subunit import Subunit

class Hyperboloid(Subunit):
    aliases = ["hyperboloid", "hourglass", "coolingtower"]

    # https://mathworld.wolfram.com/One-SheetedHyperboloid.html
    # https://www.vcalc.com/wiki/hyperboloid-volume

    def __init__(self, radius, curvature, half_height):
        """the smallest radius (at the waist), the curvature c of x^2/r^2 + y^2/r^2 - z^2/c^2 = 1, and half the height"""
        self.r,self.c,self.h = radius,curvature,half_height

    def volume(self):
        """Returns the volume of a hyperboloid"""
        return np.pi * 2*self.h * self.r**2 * ( 1 + (2*self.h)**2 / ( 12 * self.c**2 ) )

    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a hyperboloid"""
        L = 2 * self.h
        R = self.r * np.sqrt( 1 + L**2 / (4 * self.c**2 ) )
        Volume_max = 2*self.h * 2*R * 2*R
        Volume = np.pi * L * ( 2 * self.r**2 + R**2 ) / 3
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = rng.uniform(-R, R, N)
        y = rng.uniform(-R, R, N)
        z = rng.uniform(-self.h, self.h, N)
        idx = np.where(x**2/self.r**2 + y**2/self.r**2 - z**2/self.c**2 < 1.0)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the hyperboloid"""
        idx = (x_eff**2/self.r**2 + y_eff**2/self.r**2 - z_eff**2/self.c**2 > 1.0)
        return idx
