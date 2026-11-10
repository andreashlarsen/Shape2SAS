import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..helpfunctions import InputError

class Subunit(ABC):
    """
    Base class of all subunits. A subunit is a geometric body, described in
    its own frame: centred on the origin, before any rotation or translation.

    A subclass must set the class attribute
        aliases : names the user can type for the subunit; the first is the canonical name
    take its dimensions as named arguments of __init__, and implement
    getVolume(), getPointDistribution() and checkOverlap().
    """

    aliases: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs):
        """catch a missing or malformed class attribute when the subunit is defined, not when it is first used"""
        super().__init_subclass__(**kwargs)
        # aliases must be the subclass' own: inheriting them would make two subunits share the same names
        aliases = cls.__dict__.get("aliases")
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) for a in aliases):
            raise TypeError("subunit " + cls.__name__ + " must define aliases as a non-empty list of names")

    @classmethod
    def parameters(cls) -> list[str]:
        """the names of the dimensions the subunit takes, in order"""
        return list(inspect.signature(cls).parameters)

    @classmethod
    def from_dimensions(cls, dimensions):
        """
        create the subunit from a list of dimensions, in the order of its parameters, as given on the command line

        raises InputError, naming the parameters, if the wrong number of dimensions is given
        """
        dimensions = list(dimensions)
        try:
            inspect.signature(cls).bind(*dimensions)
        except TypeError:
            names = cls.parameters()
            raise InputError("subunit %s takes %d dimension%s (%s), but %d %s given: %s"
                             % (cls.aliases[0], len(names), "" if len(names) == 1 else "s", ", ".join(names),
                                len(dimensions), "was" if len(dimensions) == 1 else "were", dimensions)) from None
        return cls(*dimensions)

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
