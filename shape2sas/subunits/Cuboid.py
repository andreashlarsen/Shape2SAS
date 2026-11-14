import numpy as np 
from .Subunit import Subunit

class Cuboid(Subunit):
    aliases = ["cuboid","brick"]

    def __init__(self, side_a, side_b, side_c):
        """full side lengths along x, y and z"""
        self.a,self.b,self.c = side_a,side_b,side_c

    def volume(self):
        """Returns the volume of a cuboid"""
        return self.a * self.b * self.c
    
    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a cuboid"""
        # a, b and c are the full side lengths, so the generating box spans
        # [-a/2,a/2] etc., matching both volume() and outside()
        x_add = rng.uniform(-self.a / 2, self.a / 2, Npoints)
        y_add = rng.uniform(-self.b / 2, self.b / 2, Npoints)
        z_add = rng.uniform(-self.c / 2, self.c / 2, Npoints)
        return x_add, y_add, z_add
    
    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the cuboid"""
        idx = ((abs(x_eff) >= self.a/2) 
        | (abs(y_eff) >= self.b/2) | (abs(z_eff) >= self.c/2))
        return idx
