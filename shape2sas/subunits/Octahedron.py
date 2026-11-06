from .subunits_helpfunctions import check_dimension
from .TruncatedOctahedron import TruncatedOctahedron

class Octahedron(TruncatedOctahedron):
    """
    A general octahedron |x|/a + |y|/b + |z|/c <= 1: a truncated octahedron
    without truncation, parameterised by the vertex distances a, b and c.
    a = b = c gives a regular octahedron.
    """
    aliases = ["octahedron","octa","bipyramid"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)
        super().__init__(list(dimensions) + [0])
