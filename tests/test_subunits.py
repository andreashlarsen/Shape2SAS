#!/usr/bin/python3
"""
Self-consistency tests for the subunits, the structure factors and the
rotation machinery.
"""

import numpy as np

from shape2sas import subunits
from shape2sas.models import rotate_and_translate, undo_rotate_and_translate
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
    """getPointDistribution() and checkOverlap() must describe the same body"""
    failures = []
    for name, dimensions in CASES.items():
        obj = getattr(subunits, name)(list(dimensions))
        x, y, z = obj.getPointDistribution(N)

        # checkOverlap returns the points OUTSIDE the subunit, so a subunit's
        # own points should (almost) all be inside it. A handful may sit
        # exactly on the surface, which the >= / != tests count as outside.
        n_outside = len(obj.checkOverlap(x, y, z)[0])
        tolerance = max(10, int(0.001 * len(x)))
        if n_outside > tolerance:
            failures.append("%s: %d of %d points fall outside its own body"
                            % (name, n_outside, len(x)))

        # the sampling box and getVolume() must agree, otherwise the point
        # density - and hence the scattering - is wrong
        ratio = len(x) / N
        if not 0.85 < ratio < 1.15:
            failures.append("%s: got %d points, requested %d (ratio %.2f) - "
                            "the sampling box does not match getVolume()"
                            % (name, len(x), N, ratio))
    assert not failures, "\n".join(failures)


def test_missing_dimensions_are_rejected():
    """a subunit given the wrong number of dimensions must complain"""
    for name, dimensions in CASES.items():
        cls = getattr(subunits, name)
        try:
            cls(list(dimensions)[:-1] if len(dimensions) > 1 else [])
        except ValueError:
            continue
        raise AssertionError("%s accepted too few dimensions" % name)


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
        np.random.seed(2)
        particle = create_particle(Npoints=8000)
        for _ in range(2):
            particle.add_subunit("ellipticalcylinder", [40., 15., 90.], rotation=list(rotation))
        distribution = particle._points()
        survivors = len(distribution.x[1])
        assert survivors == 0, ("rotation %s: %d points of the buried subunit "
                                "were not excluded" % (rotation, survivors))


def test_structure_factor_with_several_subunits():
    """the decoupling approximation must cope with subunits of different sizes"""
    from shape2sas.structure_factors.structure_factors_helpfunctions import calc_A00

    np.random.seed(3)
    particle = create_particle(Npoints=2000)
    particle.add_subunit("sphere", [50.])
    particle.add_subunit("sphere", [30.], com=[60, 0, 0])
    distribution = particle._points()
    assert len(distribution.x[0]) != len(distribution.x[1]), \
        "test needs subunits with different point counts"
    A00 = calc_A00(np.linspace(0.001, 0.5, 20), distribution)
    assert np.all(np.isfinite(A00))


def test_structure_factor_names_resolve():
    """every name documented in the README must resolve to a class"""
    from shape2sas.helpfunctions import getStructureFactorClass

    documented = {
        "hardsphere": "HardSphere", "hs": "HardSphere",
        "hard-sphere": "HardSphere", "hard_sphere": "HardSphere",
        "aggregation": "Aggregation", "aggr": "Aggregation",
        "frac2d": "Aggregation", "aggregate": "Aggregation",
        "None": "NoStructure", "no": "NoStructure", "unity": "NoStructure",
    }
    for name, expected in documented.items():
        got = getStructureFactorClass(name).__name__
        assert got == expected, "%s resolved to %s, expected %s" % (name, got, expected)

    try:
        getStructureFactorClass("hardsphre")
    except ValueError:
        return
    raise AssertionError("an unknown structure factor name was accepted")


def test_structure_factor_parameter_counts():
    """a structure factor given the wrong number of parameters must complain"""
    from shape2sas import structure_factors

    for name in ("HardSphere", "Aggregation"):
        cls = getattr(structure_factors, name)
        try:
            cls([1.0])
        except ValueError:
            continue
        raise AssertionError("%s accepted the wrong number of parameters" % name)
