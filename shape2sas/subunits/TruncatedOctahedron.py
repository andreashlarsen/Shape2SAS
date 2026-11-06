import numpy as np
from .subunits_helpfunctions import check_dimension, printt

class TruncatedOctahedron:
    """
    A general octahedron with truncated vertices.

    The vertices sit on the axes at +-a, +-b and +-c, so the untruncated body
    is |x|/a + |y|/b + |z|/c <= 1. Each vertex is cut off by a square facet
    perpendicular to its axis, at a fraction 1 - t of the way out:
    |x| <= (1 - t) a, and likewise for y and z.

        t = 0    full octahedron (regular when a = b = c)
        t = 1/3  truncated octahedron (Archimedean when a = b = c)
        t = 1/2  cuboctahedron

    This is the same body as the truncated_octahedron model in SasView, with
    radius_a = a, b2a_ratio = b/a, c2a_ratio = c/a and truncation = t.
    """
    aliases = ["truncated_octahedron","truncatedocta"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,4)
        self.a,self.b,self.c,self.t = dimensions
        if not 0 <= self.t <= 0.5:
            printt("\nERROR: the truncation of a truncated_octahedron must be between 0 "
                   "and 0.5, but " + str(self.t) + " was given\n")
            exit()

    def getVolume(self):
        """Returns the volume of a truncated octahedron"""
        # each truncation removes a pyramid similar to half the octahedron,
        # scaled down by t; for t <= 1/2 the six pyramids do not overlap
        return 4 / 3 * self.a * self.b * self.c * (1 - 3 * self.t**3)

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a truncated octahedron"""
        # the square facets bound the body, so they also bound the box
        xmax, ymax, zmax = (1 - self.t) * np.array([self.a, self.b, self.c])

        Volume = self.getVolume()
        Volume_max = 2 * xmax * 2 * ymax * 2 * zmax
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-xmax, xmax, N)
        y = np.random.uniform(-ymax, ymax, N)
        z = np.random.uniform(-zmax, zmax, N)

        d = np.abs(x) / self.a + np.abs(y) / self.b + np.abs(z) / self.c
        idx = np.where(d < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a truncated octahedron"""
        d = np.abs(x_eff) / self.a + np.abs(y_eff) / self.b + np.abs(z_eff) / self.c
        idx = np.where((d >= 1) | (np.abs(x_eff) >= (1 - self.t) * self.a)
                       | (np.abs(y_eff) >= (1 - self.t) * self.b)
                       | (np.abs(z_eff) >= (1 - self.t) * self.c))

        return idx
