"""Building a point model: painting the subunits onto a static cloud of points."""

from dataclasses import dataclass, field
from typing import Optional

import itertools
import logging
import os

import numpy as np

from . import subunits
from .helpfunctions import InputError, build_alias_registry, lookup_alias

logger = logging.getLogger(__name__)

@dataclass
class ModelPointDistribution:
    """
    Point distribution of a model: the points of the static cloud that lie in
    the particle, each with the summed SLD of the subunits it lies in
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sld: np.ndarray #scattering length density for each point
    volume_total: float

@dataclass
class ModelProfile:
    """
    a particle built from subunits; the default is a sphere of radius 50 A at the origin

    subunits        : subunit names, e.g. "sphere" or "cylinder" (any alias, ignoring case, spaces, underscores and hyphens)
    dimensions      : dimensions of each subunit [A]
    sld             : excess scattering length density (contrast) of each subunit [1e-6 A^-2], default 1
    com             : displacement of each subunit [A], default (0, 0, 0)
    rotation        : rotation of each subunit, as Euler angles (intrinsic ZYX) [degrees], default (0, 0, 0)
    rotation_points : point to rotate each subunit around [A], default (0, 0, 0)
    exclude_overlap : a subunit does not paint the points of earlier subunits; otherwise, the SLDs of overlapping subunits add up
    """

    subunits: list[str] = field(default_factory=lambda: ["sphere"])
    dimensions: list[list[float]] = field(default_factory=lambda: [[50.0]])
    sld: Optional[list[float]] = None
    com: Optional[list[list[float]]] = None
    rotation: Optional[list[list[float]]] = None
    rotation_points: Optional[list[list[float]]] = None
    exclude_overlap: bool = True

# seed of the static point cloud that every model is cut out of
CLOUD_SEED = 0

# mean number of points in a cell of the static cloud
POINTS_PER_CELL = 100

def cell_size(density):
    """edge length of the cells of the static cloud at the given point density"""
    return (POINTS_PER_CELL / density) ** (1 / 3)

def cells_in_box(lower, upper, density):
    """indices of the cells of the static cloud that overlap the box [lower, upper]"""
    L = cell_size(density)
    lo = np.floor(np.asarray(lower, dtype=float) / L).astype(int)
    hi = np.floor(np.asarray(upper, dtype=float) / L).astype(int)
    return set(itertools.product(*(range(a, b + 1) for a, b in zip(lo, hi))))

def static_cloud(cells, density, seed=CLOUD_SEED):
    """
    points of the static cloud in the given cells

    The cloud is uniformly random points at the given density. To make it static,
    space is tiled into cubic cells, each filled by its own random generator,
    seeded by the cell index. A point therefore depends only on its cell: for a
    given density and seed, the cloud is the same no matter which model is cut
    out of it.
    """
    L = cell_size(density)
    points = []
    for cell in sorted(cells):
        # seeds must be non-negative, so wrap negative cell indices
        rng = np.random.default_rng([seed, *(c % 2**64 for c in cell)])
        n = rng.poisson(POINTS_PER_CELL)
        points.append((cell + rng.random((n, 3))) * L)
    return np.concatenate(points).T

class GenerateAllPoints:
    def __init__(self, Npoints, com, subunits, dimensions, rotation, sld, exclude_overlap, rotation_points=None):
        self.Npoints = Npoints
        self.com = com
        self.subunits = subunits
        self.Number_of_subunits = len(subunits)
        self.dimensions = dimensions
        self.rotation = rotation
        if rotation_points is None:
            rotation_points = [[0,0,0]] * self.Number_of_subunits
        self.rotation_points = rotation_points
        self.sld = sld
        self.exclude_overlap = exclude_overlap
        self.setAvailableSubunits()

    def setAvailableSubunits(self):
        """Dynamically build dictionary of aliases -> subunit classes"""
        self.subunitClasses = build_alias_registry(subunits)

    def getSubunitClass(self, name):
        """Look up a subunit class by any of its aliases, ignoring case,
        spaces, underscores and hyphens"""
        return lookup_alias(self.subunitClasses, name, "subunit")

    def onGeneratingAllPoints(self):
        """Paint the subunits onto the static cloud: every point of the cloud
        takes the summed SLD of the subunits it lies in"""
        subunit_objects = [
            self.getSubunitClass(self.subunits[i])(self.dimensions[i], com=self.com[i], rotation=self.rotation[i],
                                                   rotation_point=self.rotation_points[i], sld=self.sld[i])
            for i in range(self.Number_of_subunits)
        ]
        volume = [subunit.getVolume() for subunit in subunit_objects]
        sum_vol = sum(volume)
        if sum_vol <= 0:
            raise InputError("the model has no volume, so there are no points to scatter from")

        # point density that gives about Npoints points in total
        density = self.Npoints / sum_vol

        # the part of the static cloud that covers the model
        cells = set().union(*(cells_in_box(*subunit.getBoundingBox(), density) for subunit in subunit_objects))
        x, y, z = static_cloud(cells, density)

        sld = np.zeros(len(x))
        painted = np.zeros(len(x), dtype=bool)
        volume_total = 0
        for i, subunit in enumerate(subunit_objects):
            inside = subunit.contains(x, y, z)
            N_inside = np.count_nonzero(inside)
            N_shared = np.count_nonzero(inside & painted)

            # with exclude_overlap, a subunit does not paint over earlier subunits
            if self.exclude_overlap:
                inside &= ~painted
            sld[inside] += subunit.sld
            painted |= inside

            # the volume of the model is the union of its subunits
            volume_total += volume[i] * (N_inside - N_shared) / max(N_inside, 1)

            #Show information about the subunit
            logger.info(f"        {N_inside} points for subunit {i}: {self.subunits[i]}")
            logger.info(f"             Point density     : {density:.3e} (points per volume)")
            logger.info(f"             Scattering density: {density * subunit.sld:.3e} (density times scattering length)")
            if self.exclude_overlap:
                logger.info(f"             Excluded points   : {N_shared} (overlap with earlier subunits)")
            else:
                logger.info(f"             Shared points     : {N_shared} (overlap with earlier subunits, SLDs added)")

        logger.info(f"        Total points in model: {np.count_nonzero(painted)}")
        logger.info(f"        Total volume of model: {volume_total:.3e} A^3")
        logger.info(" ")

        return x[painted], y[painted], z[painted], sld[painted], volume_total

def _per_subunit(values, default, name, n_subunits):
    """one value per subunit, or the default for every subunit if no values are given"""
    if values is None:
        return [default] * n_subunits
    if len(values) != n_subunits:
        raise InputError("%d subunits, but %d values of %s were given" % (n_subunits, len(values), name))
    return list(values)

def get_point_distribution(profile: ModelProfile, Npoints: int = 3000) -> ModelPointDistribution:
    """
    cut the particle out of a static cloud of random points, at a density that gives about Npoints points in total

    the cloud depends only on the density, so the same profile and Npoints always give the same points

    raises ValueError on an unknown subunit, a wrong number of dimensions, or lists of different lengths
    """
    n = len(profile.subunits)
    dimensions = _per_subunit(profile.dimensions, None, "dimensions", n)
    sld = _per_subunit(profile.sld, 1.0, "sld", n)
    com = _per_subunit(profile.com, [0.0, 0.0, 0.0], "com", n)
    rotation = _per_subunit(profile.rotation, [0.0, 0.0, 0.0], "rotation", n)
    rotation_points = _per_subunit(profile.rotation_points, [0.0, 0.0, 0.0], "rotation_points", n)
    x_new, y_new, z_new, sld_new, volume_total = GenerateAllPoints(Npoints, com, profile.subunits, dimensions, rotation, sld, profile.exclude_overlap, rotation_points).onGeneratingAllPoints()
    return ModelPointDistribution(x=x_new, y=y_new, z=z_new, sld=sld_new, volume_total=volume_total)

def save_points(point_distribution,model_filename):
    """save point cloud to a file"""
    os.makedirs(model_filename, exist_ok=True)  
    x,y,z,sld = point_distribution.x, point_distribution.y, point_distribution.z, point_distribution.sld
    with open('%s/points_%s.txt' % (model_filename,model_filename),'w') as f:
        f.write('# x y z sld\n')
        for xi,yi,zi,s in zip(x,y,z,sld):
            f.write('%f %f %f %f\n' % (xi,yi,zi,s))

