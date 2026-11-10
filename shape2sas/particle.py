"""A particle built from subunits, and the scattering it gives."""

from collections.abc import Sequence
from typing import overload

import numpy as np

from .helpfunctions import InputError, getStructureFactorClass, getSubunitClass
from .models import ModelPointDistribution, SubunitPlacement, GenerateAllPoints
from .subunits.Subunit import Subunit
from .theoretical_scattering import PairDistanceDistribution, calc_Iq_func


class Particle:
    """
    a particle built from geometric subunits

    the particle is filled with random points, from which its pair distance distribution, p(r), and scattering are calculated.
    the points, p(r) and the form factor for the last q are calculated on first use, and kept until another subunit is added.
    create a particle with create_particle.
    """

    def __init__(self, Npoints, prpoints, polydispersity, exclude_overlap, use_ausaxs):
        self.Npoints = Npoints
        self.prpoints = prpoints
        self.polydispersity = polydispersity
        self.exclude_overlap = exclude_overlap
        self.use_ausaxs = use_ausaxs
        self._subunits = []
        self._point_distribution = None
        self._pdd = None
        self._Pq = None

    @overload
    def add_subunit(self, subunit: Subunit, /, *, sld: float = 1.0, com: Sequence[float] = (0, 0, 0),
                    rotation: Sequence[float] = (0, 0, 0), rotation_point: Sequence[float] = (0, 0, 0)) -> None:
        """add a subunit, e.g. subunits.Cylinder(radius=20, length=100), to the particle"""

    @overload
    def add_subunit(self, shape: str | type[Subunit], dimensions: Sequence[float], /, *, sld: float = 1.0,
                    com: Sequence[float] = (0, 0, 0), rotation: Sequence[float] = (0, 0, 0),
                    rotation_point: Sequence[float] = (0, 0, 0)) -> None:
        """add a subunit, given by its name, e.g. "cylinder", or class, and a list of its dimensions, e.g. [20, 100], to the particle"""

    def add_subunit(self, shape, dimensions=None, /, *, sld=1.0, com=(0, 0, 0), rotation=(0, 0, 0), rotation_point=(0, 0, 0)):
        """
        add a subunit to the particle, either as
            add_subunit(subunit, ...)            : a subunit, e.g. subunits.Cylinder(radius=20, length=100)
            add_subunit(shape, dimensions, ...)  : a subunit class, e.g. subunits.Cylinder, or its name, e.g. "cylinder" (any alias, ignoring
                                                   case, spaces, underscores and hyphens), and its dimensions [A], in the order of its
                                                   parameters, e.g. [20, 100]; see the table of subunits in the README

        sld            : excess scattering length density (contrast) [1e-6 A^-2]
        com            : displacement of the subunit [A]
        rotation       : rotation of the subunit, as Euler angles (intrinsic ZYX) [degrees]
        rotation_point : point to rotate the subunit around [A]

        raises ValueError on an unknown subunit, a wrong number of dimensions, or a vector without three components
        """
        if not isinstance(shape, Subunit):
            # the second form: build the subunit, and add it with the first
            if dimensions is None:
                raise InputError("a subunit given by name or class needs its dimensions")
            subunit = getSubunitClass(shape).from_dimensions(dimensions)
            return self.add_subunit(subunit, sld=sld, com=com, rotation=rotation, rotation_point=rotation_point)

        if dimensions is not None:
            raise InputError("the dimensions of a subunit are given when it is created, so they cannot also be given to add_subunit")
        subunit = shape
        for name, vector in (("com", com), ("rotation", rotation), ("rotation_point", rotation_point)):
            if len(vector) != 3:
                raise InputError("%s must have three components, but %s was given" % (name, list(vector)))

        self._subunits.append(SubunitPlacement(subunit, float(sld), list(com), list(rotation), list(rotation_point)))
        self._point_distribution = None
        self._pdd = None
        self._Pq = None

    def _points(self) -> ModelPointDistribution:
        if self._point_distribution is None:
            if not self._subunits:
                raise InputError("the particle has no subunits; add them with add_subunit")
            self._point_distribution = GenerateAllPoints(self.Npoints, self._subunits, self.exclude_overlap).onGeneratingAllPointsSeparately()
        return self._point_distribution

    def _pair_distances(self) -> PairDistanceDistribution:
        if self._pdd is None:
            self._pdd = PairDistanceDistribution(self._points(), self.prpoints, self.polydispersity, self.use_ausaxs)
        return self._pdd

    def _form_factor(self, q):
        """form factor normalized so P(0) = 1, kept for the last q, since the structure factor and the scattering both need it"""
        if self._Pq is None or not np.array_equal(self._Pq[0], q):
            self._Pq = (q.copy(), self._pair_distances().calc_Pq(q))
        return self._Pq[1]

    def points(self):
        """the points filling the particle, as arrays x, y, z [A] and their sld [1e-6 A^-2]"""
        p = self._points()
        return np.concatenate(p.x), np.concatenate(p.y), np.concatenate(p.z), np.concatenate(p.sld)

    @property
    def volume(self):
        """volume of the particle, without the overlap between subunits if that is excluded [A^3]"""
        return self._points().volume_total

    def pr(self):
        """pair distance distribution, as arrays r [A] and p(r), normalized so max p(r) = 1"""
        pdd = self._pair_distances()
        return pdd.r, pdd.pr

    @property
    def dmax(self):
        """maximum distance in the particle [A]"""
        return self._pair_distances().dmax

    @property
    def Rg(self):
        """radius of gyration [A]"""
        return self._pair_distances().Rg

    def structure_factor(self, q, name="none", parameters=()):
        """
        effective structure factor, S_eff(q), of the particles (decoupling approximation)

        q          : q values [1/A]
        name       : "none", "hardsphere" or "aggregation" (or any of their aliases)
        parameters : parameters of the structure factor, see its par_names
                     (hardsphere: concentration, R_HS; aggregation: R_eff, N_aggr, fraction_aggregated)

        raises ValueError on an unknown structure factor or a wrong number of its parameters
        """
        q = np.asarray(q, dtype=float)
        return self._structure_factor(q, self._form_factor(q), name, parameters)

    def _structure_factor(self, q, Pq_norm, name, parameters):
        # the decoupling approximation needs the form factor normalized so P(0) = 1
        return getStructureFactorClass(name)(parameters).structure_eff(q, self._points(), Pq_norm)

    def scattering(self, q, conc=0.02, structure_factor="none", structure_factor_parameters=(), sigma_r=0.0):
        """
        scattering of the particles in absolute units [cm^-1], I(q) = conc * P(q) * S_eff(q) * exp(-(q sigma_r)^2 / 2),
        where the form factor P(q) is per unit volume fraction, so P(0) = V * contrast^2

        q                           : q values [1/A]
        conc                        : volume fraction of the particles
        structure_factor            : see structure_factor
        structure_factor_parameters : see structure_factor
        sigma_r                     : interface roughness [A]

        raises ValueError on an unknown structure factor or a wrong number of its parameters
        """
        q = np.asarray(q, dtype=float)
        Pq_norm = self._form_factor(q)
        S_eff = self._structure_factor(q, Pq_norm, structure_factor, structure_factor_parameters)
        return conc * self._pair_distances().calc_P0() * calc_Iq_func(q, Pq_norm, S_eff, sigma_r)


def create_particle(Npoints=3000, prpoints=100, polydispersity=0.0, exclude_overlap=True, use_ausaxs=True) -> Particle:
    """
    create an empty particle; add its subunits with add_subunit

    Npoints         : roughly how many random points fill the particle, split between the subunits by their volume
    prpoints        : number of bins in p(r)
    polydispersity  : relative polydispersity of the particle size
    exclude_overlap : remove the points of a subunit that lie inside earlier subunits
    use_ausaxs      : try the AUSAXS backend first, and fall back to the default method if pyausaxs is not installed or fails
    """
    return Particle(Npoints, prpoints, polydispersity, exclude_overlap, use_ausaxs)
