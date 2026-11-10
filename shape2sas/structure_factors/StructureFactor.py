from abc import ABC, abstractmethod

import numpy as np

from ..aliased import Aliased
from ..sesans.sesans_helpfunctions import default_sesans_range
from .structure_factors_helpfunctions import decoupling_approx

class StructureFactor(Aliased, ABC):
    """
    Base class of all structure factors.

    A subclass must set the class attribute
        aliases : names the user can type after --S; the first is the canonical name
    take its parameters as named arguments of __init__, and implement structure_factor().
    """

    _kind = "structure factor"
    _value_name = "parameter"

    @classmethod
    def from_parameters(cls, parameters):
        """
        create the structure factor from a list of parameters, in the order of its arguments, as given on the command line

        raises InputError, naming the parameters, if the wrong number of parameters is given
        """
        return cls._from_values(parameters)

    @abstractmethod
    def structure_factor(self, q: np.ndarray) -> np.ndarray:
        """Returns the structure factor S(q) of spherical, monodisperse particles"""

    def effective_structure_factor(self, q, point_distribution, Pq):
        """
        Returns the effective structure factor S_eff(q) of the particles in point_distribution, with form factor Pq (normalised so P(0) = 1)

        The decoupling approximation corrects S(q) for particles that are not spherical (or that are polydisperse). Override this
        only if the structure factor already accounts for that.
        """
        return decoupling_approx(q, point_distribution, Pq, self.structure_factor(q))

    def sesans_range(self, dmax):
        """
        Returns (qmin, deltamax) for a SESANS calculation of particles with maximum distance dmax

        Override this only if the structure factor introduces a length scale larger than the particle itself (as aggregation
        does); otherwise the default, set by dmax, is the right choice.
        """
        return default_sesans_range(dmax)
