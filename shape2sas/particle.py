"""A particle built from subunits, the random points filling it, and the scattering they give."""

from collections.abc import Sequence
from functools import cached_property
from typing import overload

import numpy as np

from .helpfunctions import InputError, getStructureFactorClass, getSubunitClass
from .models import ModelPointDistribution, SubunitPlacement, GenerateAllPoints
from .subunits.Subunit import Subunit
from .theoretical_scattering import PairDistanceDistribution, calc_Iq_func


class PointDistribution:
    """
    the random points filling a particle, drawn by Particle.generate_points, and the scattering calculated from them

    the points never change: other subunits or other settings need a new point distribution from generate_points.
    the histogram is therefore calculated on first use and kept, as is the form factor for the last q.

    attributes:
    x, y, z        : coordinates of the points [A], read-only
    sld            : excess scattering length density (contrast) of each point [1e-6 A^-2], read-only
    volume         : volume of the particle, without the overlap between subunits if that is excluded [A^3]
    prpoints       : number of bins in p(r)
    polydispersity : relative polydispersity of the particle size
    use_ausaxs     : whether the AUSAXS backend is tried first for the histogram
    """

    def __init__(self, distribution: ModelPointDistribution, prpoints, polydispersity, use_ausaxs):
        self._distribution = distribution
        self.x, self.y, self.z, self.sld = (np.concatenate(a) for a in (distribution.x, distribution.y, distribution.z, distribution.sld))
        for a in (self.x, self.y, self.z, self.sld):
            a.setflags(write=False)
        self.volume = distribution.volume_total
        self.prpoints = prpoints
        self.polydispersity = polydispersity
        self.use_ausaxs = use_ausaxs
        self._Pq = None

    @cached_property
    def histogram(self) -> PairDistanceDistribution:
        """
        the pair distance distribution of the points, with
            r, pr      : p(r), with roughly prpoints bins, normalized so max p(r) = 1
            r_hist, hr : full-resolution h(r)
            dmax, Rg   : maximum distance and radius of gyration [A]
            calc_Pq(q) : form factor, normalized so P(0) = 1
            calc_P0()  : forward scattering per unit volume fraction, V * contrast^2
        """
        return PairDistanceDistribution(self._distribution, self.prpoints, self.polydispersity, self.use_ausaxs)

    def _form_factor(self, q):
        """form factor normalized so P(0) = 1, kept for the last q, since the structure factor and the scattering both need it"""
        if self._Pq is None or not np.array_equal(self._Pq[0], q):
            self._Pq = (q.copy(), self.histogram.calc_Pq(q))
        return self._Pq[1]

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
        return getStructureFactorClass(name)(parameters).structure_eff(q, self._distribution, Pq_norm)

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
        return conc * self.histogram.calc_P0() * calc_Iq_func(q, Pq_norm, S_eff, sigma_r)


class Particle:
    """
    a particle built from geometric subunits

    give its subunits to create_particle, or add them with add_subunit, then fill it with random points with generate_points, which gives the pair distance
    distribution and the scattering. create a particle with create_particle.
    """

    def __init__(self, exclude_overlap):
        self.exclude_overlap = exclude_overlap
        self._subunits = []

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

        point distributions generated before the subunit is added are not affected.
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

    def generate_points(self, Npoints: int, *, prpoints: int = 100, polydispersity: float = 0.0, use_ausaxs: bool = True) -> PointDistribution:
        """
        fill the particle with random points

        Npoints        : roughly how many points fill the particle, split between the subunits by their volume
        prpoints       : number of bins in p(r)
        polydispersity : relative polydispersity of the particle size
        use_ausaxs     : try the AUSAXS backend first for the histogram, and fall back to the default method if pyausaxs is not
                         installed or fails

        every call draws new points. raises ValueError if the particle has no subunits
        """
        if not self._subunits:
            raise InputError("the particle has no subunits; add them with add_subunit")
        distribution = GenerateAllPoints(Npoints, list(self._subunits), self.exclude_overlap).onGeneratingAllPointsSeparately()
        return PointDistribution(distribution, prpoints, polydispersity, use_ausaxs)


def create_particle(*subunits: Subunit | Sequence, exclude_overlap: bool = True) -> Particle:
    """
    create a particle from any number of subunits; more can be added with add_subunit. fill it with points with generate_points

    subunits        : each is forwarded to add_subunit, as
                        a subunit                                   e.g. subunits.Cylinder(radius=20, length=100)
                        a list of the arguments of add_subunit      e.g. ["cylinder", [20, 100]] or [subunits.Cylinder(20, 100)]
                      where the list may end with a dict of the keyword arguments of add_subunit, e.g. {"com": [0, 0, 50]}
    exclude_overlap : remove the points of a subunit that lie inside earlier subunits

    raises ValueError on a subunit that add_subunit does not accept
    """
    particle = Particle(exclude_overlap)
    for subunit in subunits:
        if isinstance(subunit, Subunit):
            particle.add_subunit(subunit)
            continue
        if isinstance(subunit, (str, bytes)) or not isinstance(subunit, Sequence) or not subunit:
            raise InputError("a subunit must be given as a subunit, or as a list of the arguments of add_subunit, but %r was given"
                             % (subunit,))
        args = list(subunit)
        kwargs = args.pop() if isinstance(args[-1], dict) else {}
        particle.add_subunit(*args, **kwargs)
    return particle
