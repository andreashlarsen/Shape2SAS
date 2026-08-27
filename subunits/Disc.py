from .EllipticalCylinder import EllipticalCylinder

class Disc(EllipticalCylinder):
    """An elliptical disc: an elliptical cylinder, parameterised by the two
    semi-axes a, b and the length l.

    Kept distinct from CircularDisc (R, l) so that 'disc' means the same thing
    here as it does in the SasView integration.
    """
    aliases = ["disc","disk","ellipticaldisc"]
