"""A particle built from subunits, the random points filling it, and the scattering they give."""

from collections.abc import Sequence
from functools import cached_property
from typing import overload

import numpy as np

from .helpfunctions import InputError, getSubunitClass
from .models import ModelPointDistribution, GenerateAllPoints
from .structure_factors import NoStructure, StructureFactor
from .subunits.Subunit import Placement, Subunit
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
            r, pr                : p(r), with roughly prpoints bins, normalized so max p(r) = 1
            r_hist, hr           : full-resolution h(r)
            dmax, Rg             : maximum distance and radius of gyration [A]
            form_factor(q)       : form factor, normalized so P(0) = 1
            forward_scattering() : forward scattering per unit volume fraction, V * contrast^2 [cm^-1]
        """
        return PairDistanceDistribution(self._distribution, self.prpoints, self.polydispersity, self.use_ausaxs)

    def _form_factor(self, q):
        """form factor normalized so P(0) = 1, kept for the last q, since the structure factor and the scattering both need it"""
        if self._Pq is None or not np.array_equal(self._Pq[0], q):
            self._Pq = (q.copy(), self.histogram.form_factor(q))
        return self._Pq[1]

    def effective_structure_factor(self, q, structure_factor: StructureFactor | None = None):
        """
        effective structure factor, S_eff(q), of the particles (decoupling approximation)

        q                : q values [1/A]
        structure_factor : e.g. structure_factors.HardSphere(volume_fraction=0.2, radius=50); None for dilute particles, S = 1
        """
        q = np.asarray(q, dtype=float)
        return self._effective_structure_factor(q, self._form_factor(q), structure_factor)

    def _effective_structure_factor(self, q, Pq_norm, structure_factor):
        if structure_factor is None:
            structure_factor = NoStructure()
        if not isinstance(structure_factor, StructureFactor):
            raise InputError("a structure factor must be given as e.g. structure_factors.HardSphere(volume_fraction=0.2, radius=50), "
                             "but %r was given" % (structure_factor,))
        # the decoupling approximation needs the form factor normalized so P(0) = 1
        return structure_factor.effective_structure_factor(q, self._distribution, Pq_norm)

    def scattering(self, q, *, conc: float = 0.02, structure_factor: StructureFactor | None = None, sigma_r: float = 0.0):
        """
        scattering of the particles in absolute units [cm^-1], I(q) = conc * P(q) * S_eff(q) * exp(-(q sigma_r)^2 / 2),
        where the form factor P(q) is per unit volume fraction, so P(0) = V * contrast^2

        q                : q values [1/A]
        conc             : volume fraction of the particles
        structure_factor : see effective_structure_factor
        sigma_r          : interface roughness [A]
        """
        q = np.asarray(q, dtype=float)
        Pq_norm = self._form_factor(q)
        S_eff = self._effective_structure_factor(q, Pq_norm, structure_factor)
        return conc * self.histogram.forward_scattering() * calc_Iq_func(q, Pq_norm, S_eff, sigma_r)


class Particle:
    """
    a particle built from geometric subunits

    give its subunits to create_particle, or add them with add_subunit, then fill it with random points with generate_points,
    which gives the pair distance distribution and the scattering.
    """

    def __init__(self, exclude_overlap):
        self.exclude_overlap = exclude_overlap
        self._placements: list[Placement] = []

    @overload
    def add_subunit(self, subunit: Subunit | Placement, /) -> None:
        """add a subunit, e.g. subunits.Cylinder(radius=20, length=100), or a placed one, e.g. Cylinder(...).placed(com=(0, 0, 50))"""

    @overload
    def add_subunit(self, shape: str | type[Subunit], dimensions: Sequence[float], /, *, sld: float = 1.0,
                    com: Sequence[float] = (0, 0, 0), rotation: Sequence[float] = (0, 0, 0),
                    rotation_point: Sequence[float] = (0, 0, 0)) -> None:
        """add a subunit, given by its name, e.g. "cylinder", or class, and a list of its dimensions, e.g. [20, 100], and place it"""

    def add_subunit(self, shape, dimensions=None, /, **placement):
        """
        add a subunit to the particle, either as
            add_subunit(subunit)                 : a subunit, e.g. subunits.Cylinder(radius=20, length=100), at the origin with unit
                                                   contrast, or a placed subunit, e.g. subunits.Cylinder(...).placed(com=(0, 0, 50))
            add_subunit(shape, dimensions, ...)  : a subunit class, e.g. subunits.Cylinder, or its name, e.g. "cylinder" (any alias,
                                                   ignoring case, spaces, underscores and hyphens), and its dimensions [A], in the order of
                                                   its parameters, e.g. [20, 100] (see the table of subunits in the README), placed by the
                                                   keyword arguments of Subunit.placed: sld, com, rotation and rotation_point

        point distributions generated before the subunit is added are not affected.
        raises ValueError on an unknown subunit, a wrong number of dimensions, or a vector without three components
        """
        if isinstance(shape, (Subunit, Placement)):
            if dimensions is not None or placement:
                raise InputError("a subunit is created with its dimensions and placed with its placed() method, "
                                 "e.g. Cylinder(radius=20, length=100).placed(com=(0, 0, 50)), so add_subunit takes nothing else")
            self._placements.append(shape if isinstance(shape, Placement) else shape.placed())
            return

        # the second form: build the subunit and its placement, and add them with the first
        if dimensions is None:
            raise InputError("a subunit given by name or class needs its dimensions")
        self.add_subunit(getSubunitClass(shape).from_dimensions(dimensions).placed(**placement))

    def generate_points(self, Npoints: int, *, rng: np.random.Generator | int | None = None, prpoints: int = 100,
                        polydispersity: float = 0.0, use_ausaxs: bool = True) -> PointDistribution:
        """
        fill the particle with random points

        Npoints        : roughly how many points fill the particle, split between the subunits by their volume
        rng            : random number generator (np.random.Generator), or a seed for one; None draws a fresh seed
        prpoints       : number of bins in p(r)
        polydispersity : relative polydispersity of the particle size
        use_ausaxs     : try the AUSAXS backend first for the histogram, and fall back to the default method if pyausaxs is not
                         installed or fails

        every call draws new points, unless the same seed is given. raises ValueError if the particle has no subunits
        """
        if not self._placements:
            raise InputError("the particle has no subunits; add them with add_subunit")
        distribution = GenerateAllPoints(Npoints, list(self._placements), self.exclude_overlap,
                                         np.random.default_rng(rng)).onGeneratingAllPointsSeparately()
        return PointDistribution(distribution, prpoints, polydispersity, use_ausaxs)


def create_particle(*subunits: Subunit | Placement, exclude_overlap: bool = True) -> Particle:
    """
    create a particle from any number of subunits; more can be added with add_subunit. fill it with points with generate_points

    subunits        : subunits, e.g. subunits.Cylinder(radius=20, length=100), at the origin with unit contrast, or
                      placed subunits, e.g. subunits.Cylinder(radius=20, length=100).placed(com=(0, 0, 50), sld=2)
    exclude_overlap : remove the points of a subunit that lie inside earlier subunits

    raises ValueError on anything else
    """
    particle = Particle(exclude_overlap)
    for subunit in subunits:
        if not isinstance(subunit, (Subunit, Placement)):
            raise InputError("create_particle takes subunits, e.g. Sphere(radius=50), or placed subunits, e.g. "
                             "Sphere(radius=50).placed(com=(0, 0, 60)), but %r was given; "
                             "a subunit given by name is added with add_subunit" % (subunit,))
        particle.add_subunit(subunit)
    return particle
