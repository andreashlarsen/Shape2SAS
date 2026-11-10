from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..aliased import Aliased
from ..helpfunctions import InputError

class Subunit(Aliased, ABC):
    """
    Base class of all subunits. A subunit is a geometric body, described in
    its own frame: centred on the origin, before any rotation or translation.

    A subclass must set the class attribute
        aliases : names the user can type for the subunit; the first is the canonical name
    take its dimensions as named arguments of __init__, and implement
    volume(), generate_points() and outside().
    """

    _kind = "subunit"
    _value_name = "dimension"

    @classmethod
    def from_dimensions(cls, dimensions):
        """
        create the subunit from a list of dimensions, in the order of its parameters, as given on the command line

        raises InputError, naming the parameters, if the wrong number of dimensions is given
        """
        return cls._from_values(dimensions)

    def placed(self, *, sld: float = 1.0, com: Sequence[float] = (0, 0, 0), rotation: Sequence[float] = (0, 0, 0),
               rotation_point: Sequence[float] = (0, 0, 0)) -> "Placement":
        """
        the subunit, placed in a particle

        sld            : excess scattering length density (contrast) [1e-6 A^-2]
        com            : displacement of the subunit [A]
        rotation       : rotation of the subunit, as Euler angles (intrinsic ZYX) [degrees]
        rotation_point : point to rotate the subunit around, in the subunit's own frame [A]

        raises InputError on a vector without three components
        """
        return Placement(self, sld, com, rotation, rotation_point)

    @abstractmethod
    def volume(self) -> float:
        """Returns the volume of the subunit itself - not of the box used to generate its points"""

    @abstractmethod
    def generate_points(self, Npoints: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns x, y and z of (about) Npoints points, drawn with rng uniformly in the subunit"""

    @abstractmethod
    def outside(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Returns a boolean array, true for the points OUTSIDE the subunit.

        Points are given in the subunit's own frame (rotation and translation
        already undone by the caller).
        """


@dataclass(frozen=True)
class Placement:
    """a subunit, and where it is placed in a particle; create it with Subunit.placed"""

    subunit: Subunit
    sld: float = 1.0
    com: tuple[float, float, float] = (0., 0., 0.)
    rotation: tuple[float, float, float] = (0., 0., 0.)
    rotation_point: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if not isinstance(self.subunit, Subunit):
            raise InputError("a placement needs a subunit, but %r was given" % (self.subunit,))
        object.__setattr__(self, "sld", float(self.sld))
        for name in ("com", "rotation", "rotation_point"):
            vector = getattr(self, name)
            if len(vector) != 3:
                raise InputError("%s must have three components, but %s was given" % (name, list(vector)))
            object.__setattr__(self, name, tuple(float(v) for v in vector))
