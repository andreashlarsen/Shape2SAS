import numpy as np
from .subunits_helpfunctions import check_dimension

class Octahedron:
    aliases = ["octahedron","octa","bipyramid"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,1)
        self.a = dimensions[0]
        # a is the edge length; the six vertices sit on the axes at +-r,
        # so the body is |x| + |y| + |z| <= r
        self.r = self.a / np.sqrt(2)

    def getVolume(self):
        """Returns the volume of a regular octahedron"""
        return np.sqrt(2) / 3 * self.a**3

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a regular octahedron"""
        Volume = self.getVolume()
        Volume_max = (2 * self.r)**3
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.r, self.r, N)
        y = np.random.uniform(-self.r, self.r, N)
        z = np.random.uniform(-self.r, self.r, N)

        d = np.abs(x) + np.abs(y) + np.abs(z)
        idx = np.where(d < self.r)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a regular octahedron"""
        d = np.abs(x_eff) + np.abs(y_eff) + np.abs(z_eff)
        idx = np.where(d >= self.r)

        return idx
