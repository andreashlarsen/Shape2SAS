"""A particle built from geometric subunits."""

import numpy as np

from ..helpfunctions import InputError
from ..subunits.Subunit import Placement, Subunit
from .GenerateAllPoints import GenerateAllPoints
from .PointDistribution import PointDistribution


class Particle:
    """
    a particle built from geometric subunits

    give its subunits to create_particle, or add them with add_subunit, then fill it with random points with generate_points,
    which gives the pair distance distribution and the scattering.
    """

    def __init__(self, exclude_overlap):
        self.exclude_overlap = exclude_overlap
        self._placements: list[Placement] = []

    def add_subunit(self, subunit: Subunit | Placement, /) -> None:
        """
        add a subunit to the particle: a subunit, e.g. subunits.Cylinder(radius=20, length=100), at the origin with unit contrast,
        or a placed subunit, e.g. subunits.Cylinder(radius=20, length=100).placed(com=(0, 0, 50), sld=2)

        point distributions generated before the subunit is added are not affected. raises ValueError on anything else
        """
        if not isinstance(subunit, (Subunit, Placement)):
            raise InputError("add_subunit takes a subunit, e.g. Sphere(radius=50), or a placed subunit, e.g. "
                             "Sphere(radius=50).placed(com=(0, 0, 60)), but %r was given" % (subunit,))
        self._placements.append(subunit if isinstance(subunit, Placement) else subunit.placed())

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
        particle.add_subunit(subunit)
    return particle
