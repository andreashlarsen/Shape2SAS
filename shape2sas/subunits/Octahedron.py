from .Subunit import Subunit
from .TruncatedOctahedron import TruncatedOctahedron

class Octahedron(TruncatedOctahedron):
    """
    A general octahedron |x|/a + |y|/b + |z|/c <= 1: a truncated octahedron without truncation, parameterised by the vertex distances
    a, b and c. a = b = c gives a regular octahedron.
    """
    aliases = ["octahedron","octa","bipyramid"]
    n_dimensions = 3

    def __init__(self, dimensions):
        # skip TruncatedOctahedron.__init__, which expects the truncation as a fourth dimension
        Subunit.__init__(self, dimensions)
        self.a,self.b,self.c = dimensions
        self.t = 0
