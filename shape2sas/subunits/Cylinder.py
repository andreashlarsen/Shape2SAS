import numpy as np 
from .Subunit import Subunit

class Cylinder(Subunit):
    aliases = ["cylinder","rod","cyl"]

    def __init__(self, radius, length):
        self.R,self.l = radius,length

    def volume(self):
        """Returns the volume of a cylinder"""

        return np.pi * self.R**2 * self.l
    
    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a cylinder"""

        Volume = self.volume()
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = rng.uniform(-self.R, self.R, N)
        y = rng.uniform(-self.R, self.R, N)
        z = rng.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where(d < self.R)
        x_add,y_add,z_add = x[idx],y[idx],z[idx]

        return x_add, y_add, z_add

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the cylinder"""

        d = np.sqrt(x_eff**2+y_eff**2)
        idx = ((d > self.R) | (abs(z_eff) > self.l / 2))
        return idx

