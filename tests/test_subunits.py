#!/usr/bin/python3
"""
Self-consistency tests for the subunits, the structure factors and the
rotation machinery.
"""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

from shape2sas import structure_factors, subunits
from shape2sas.cli_models import alias_registry, make_structure_factor, make_subunit, normalise_alias, parameter_names
from shape2sas.rotation import rotate_and_translate, undo_rotate_and_translate
from shape2sas.particle import create_particle

# one representative set of dimensions per subunit
CASES = {
    "Sphere":             [50.],
    "Cube":               [50.],
    "Cuboid":             [50., 40., 30.],
    "Octahedron":         [40., 30., 25.],
    "TruncatedOctahedron": [40., 30., 25., 0.3],
    "Cylinder":           [30., 80.],
    "CircularDisc":       [30., 20.],
    "CylinderRing":       [50., 30., 40.],
    "Ellipsoid":          [50., 30., 20.],
    "EllipticalCylinder": [40., 20., 60.],
    "Disc":               [40., 20., 15.],
    "HollowCube":         [50., 30.],
    "HollowSphere":       [50., 30.],
    "Superellipsoid":     [50., 1.0, 2.0, 2.0],
    "Ellipsoid_shell":    [50., 40., 30., 5.],
    "Hyperboloid":        [30., 40., 25.],
    "Torus":              [50., 15.],
}

N = 40000
ROTATIONS = [(0, 0, 0), (30, 0, 0), (0, 40, 0), (0, 0, 50), (30, 40, 50), (11, -73, 145)]


def test_subunits_are_self_consistent():
    """generate_points() and outside() must describe the same body"""
    failures = []
    rng = np.random.default_rng(0)
    for name, dimensions in CASES.items():
        obj = getattr(subunits, name)(*dimensions)
        x, y, z = obj.generate_points(N, rng)

        # outside() is true for the points OUTSIDE the subunit, so a subunit's
        # own points should (almost) all be inside it. A handful may sit
        # exactly on the surface, which the >= / != tests count as outside.
        n_outside = int(np.count_nonzero(obj.outside(x, y, z)))
        tolerance = max(10, int(0.001 * len(x)))
        if n_outside > tolerance:
            failures.append("%s: %d of %d points fall outside its own body"
                            % (name, n_outside, len(x)))

        # the sampling box and volume() must agree, otherwise the point
        # density - and hence the scattering - is wrong
        ratio = len(x) / N
        if not 0.85 < ratio < 1.15:
            failures.append("%s: got %d points, requested %d (ratio %.2f) - "
                            "the sampling box does not match volume()"
                            % (name, len(x), N, ratio))
    assert not failures, "\n".join(failures)


def test_named_and_listed_dimensions_give_the_same_subunit():
    """keywords, positional arguments and the command line's name and list are three spellings of the same subunit"""
    for name, dimensions in CASES.items():
        cls = getattr(subunits, name)
        keywords = dict(zip(parameter_names(cls), dimensions))
        for obj in (cls(*dimensions), make_subunit(cls.aliases[-1], dimensions)):
            assert type(obj) is cls, name
            assert obj.volume() == cls(**keywords).volume(), name


def test_structure_factor_without_parameters_ignores_placeholders():
    """on the command line, -Sp is matched to the models by position, so -S None HS -Sp 0 -Sp 0.1,60 must work"""
    assert type(make_structure_factor("None", [0.])) is structure_factors.NoStructure
    assert type(make_structure_factor("None", [])) is structure_factors.NoStructure
    # a structure factor with parameters still needs the right number
    with pytest.raises(ValueError):
        make_structure_factor("HS", [0.1])


def test_rotation_round_trip():
    """undo_rotate_and_translate() must invert rotate_and_translate()"""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-50, 50, (3, 2000))
    for rotation in ROTATIONS:
        for rotation_point, com in ([[0, 0, 0], [0, 0, 0]],
                                    [[10, -5, 3], [60, 0, -20]]):
            moved = rotate_and_translate(*pts, rotation, rotation_point, com)
            back = undo_rotate_and_translate(*moved, rotation, rotation_point, com)
            err = np.max(np.abs(np.array(back) - pts))
            assert err < 1e-9, ("rotation %s (around %s, com %s) does not round "
                                "trip: max error %.3e" % (rotation, rotation_point, com, err))


def test_buried_subunit_is_fully_excluded():
    """a subunit placed inside an identical one must be entirely excluded

    An asymmetric subunit is used, because a symmetric one hides errors in the
    inverse rotation.
    """
    for rotation in ROTATIONS:
        body = subunits.EllipticalCylinder(a=40., b=15., length=90.).placed(rotation=rotation)
        particle = create_particle(body, body)
        # the points of each subunit are only kept apart internally
        distribution = particle.generate_points(8000, rng=2)._distribution
        survivors = len(distribution.x[1])
        assert survivors == 0, ("rotation %s: %d points of the buried subunit "
                                "were not excluded" % (rotation, survivors))


def test_structure_factor_with_several_subunits():
    """the decoupling approximation must cope with subunits of different sizes"""
    from shape2sas.structure_factors.structure_factors_helpfunctions import calc_A00

    particle = create_particle(subunits.Sphere(radius=50.), subunits.Sphere(radius=30.).placed(com=(60, 0, 0)))
    # the points of each subunit are only kept apart internally
    distribution = particle.generate_points(2000, rng=3)._distribution
    assert len(distribution.x[0]) != len(distribution.x[1]), \
        "test needs subunits with different point counts"
    A00 = calc_A00(np.linspace(0.001, 0.5, 20), distribution)
    assert np.all(np.isfinite(A00))