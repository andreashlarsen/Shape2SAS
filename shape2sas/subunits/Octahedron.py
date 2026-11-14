from .TruncatedOctahedron import TruncatedOctahedron

class Octahedron(TruncatedOctahedron):
    """
    A general octahedron |x|/a + |y|/b + |z|/c <= 1: a truncated octahedron without truncation, parameterised by the vertex distances
    a, b and c. a = b = c gives a regular octahedron.
    """
    aliases = ["octahedron","octa","bipyramid"]

    def __init__(self, a, b, c):
        super().__init__(a, b, c, truncation=0)
