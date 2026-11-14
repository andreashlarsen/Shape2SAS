from .EllipticalCylinder import EllipticalCylinder

class Disc(EllipticalCylinder):
    """
    An elliptical disc: an elliptical cylinder, parameterised by the two
    semi-axes a, b of its cross section and its length.
    """
    aliases = ["disc","disk","ellipticaldisc"]
