from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .subunits_helpfunctions import check_dimension

class Subunit(ABC):
    """
    Base class of all subunits. A subunit is a geometric body, described in
    its own frame: centred on the origin, before any rotation or translation.

    A subclass must set the class attributes
        aliases      : names the user can type for the subunit; the first is the canonical name
        n_dimensions : number of dimensions the subunit takes
    call super().__init__(dimensions) from its __init__, and implement
    getVolume(), getPointDistribution() and checkOverlap().
    """

    aliases: ClassVar[list[str]]
    n_dimensions: ClassVar[int]

    def __init_subclass__(cls, **kwargs):
        """catch a missing or malformed class attribute when the subunit is defined, not when it is first used"""
        super().__init_subclass__(**kwargs)
        # aliases must be the subclass' own: inheriting them would make two subunits share the same names
        aliases = cls.__dict__.get("aliases")
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) for a in aliases):
            raise TypeError("subunit " + cls.__name__ + " must define aliases as a non-empty list of names")
        if not isinstance(getattr(cls, "n_dimensions", None), int):
            raise TypeError("subunit " + cls.__name__ + " must define n_dimensions as an integer")

    def __init__(self, dimensions):
        check_dimension(self.aliases[0], dimensions, self.n_dimensions)

    @abstractmethod
    def getVolume(self) -> float:
        """Returns the volume of the subunit itself - not of the box used to generate its points"""

    @abstractmethod
    def getPointDistribution(self, Npoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns x, y and z of (about) Npoints points, uniformly distributed in the subunit"""

    @abstractmethod
    def checkOverlap(self, x_eff: np.ndarray, y_eff: np.ndarray, z_eff: np.ndarray) -> tuple[np.ndarray, ...]:
        """Returns the indices, as given by np.where, of the points OUTSIDE the subunit.

        Points are given in the subunit's own frame (rotation and translation
        already undone by the caller).
        """
