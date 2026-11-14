import numpy as np 
from .Subunit import Subunit

class HollowSphere(Subunit):
    aliases = ["hollowsphere","shell"]

    def __init__(self, radius, inner_radius):
        self.R,self.r = radius,inner_radius

    def volume(self):
        """Returns the volume of a hollow sphere"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 4 * np.pi * self.R**2 #surface area of a sphere
        else: 
            return (4 / 3) * np.pi * (self.R**3 - self.r**3)

    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a hollow sphere"""
        Volume = self.volume()
        if self.r == self.R:
            #The hollow sphere is a shell if r=R
            phi = rng.uniform(0,2 * np.pi, Npoints)
            costheta = rng.uniform(-1, 1, Npoints)
            theta = np.arccos(costheta)

            x_add = self.R * np.sin(theta) * np.cos(phi)
            y_add = self.R * np.sin(theta) * np.sin(phi)
            z_add = self.R * np.cos(theta)
            return x_add, y_add, z_add
        Volume_max = (2*self.R)**3
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = rng.uniform(-self.R, self.R, N)
        y = rng.uniform(-self.R, self.R, N)
        z = rng.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the hollow sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        if self.r > self.R:
             self.r, self.R = self.R, self.r
        if self.r == self.R:
            idx = (d != self.R)
            return idx
        else:
            idx = ((d > self.R) | (d < self.r))
            return idx
