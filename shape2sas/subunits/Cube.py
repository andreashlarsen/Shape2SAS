import numpy as np 
from .Subunit import Subunit

class Cube(Subunit):
    aliases = ["cube","dice"]

    def __init__(self, side):
        self.a = side

    def volume(self):
        """Returns the volume of a cube"""
        return self.a**3

    def generate_points(self, Npoints, rng):
        """Returns points drawn uniformly in a cube"""
        # a is the full side length, so the generating box IS the cube:
        # no rejection needed, and the box volume matches volume()
        N = Npoints
        x_add = rng.uniform(-self.a / 2, self.a / 2, N)
        y_add = rng.uniform(-self.a / 2, self.a / 2, N)
        z_add = rng.uniform(-self.a / 2, self.a / 2, N)

        return x_add, y_add, z_add

    def outside(self, x_eff, y_eff, z_eff):
        """Returns which points are outside the cube"""
        idx = ((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2))
        
        return idx
