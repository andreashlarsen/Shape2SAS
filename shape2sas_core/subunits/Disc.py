from .EllipticalCylinder import EllipticalCylinder

class Disc(EllipticalCylinder):
    """
    An elliptical disc: an elliptical cylinder, parameterised by the two
    semi-axes a, b and the length l.
    """
    aliases = ["disc","disk","ellipticaldisc"]
