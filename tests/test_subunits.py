#!/usr/bin/python3
"""
Self-consistency tests for the subunits, the structure factors and the
rotation machinery.

Run with:  python test_subunits.py

These check the invariants that the point sampling relies on:

  1. every point returned by getPointDistribution() lies inside the body that
     checkOverlap() describes - the two must describe the SAME shape
  2. the number of points returned is close to the number requested, i.e.
     getVolume() and the sampling box agree
  3. rotating a set of points and then undoing the rotation is the identity,
     for arbitrary combinations of the three Euler angles
  4. a subunit that is completely buried inside another one is completely
     removed by the overlap exclusion
  5. every structure factor name given in the README resolves, and an
     unrecognised one is an error rather than a silent S(q) = 1
"""

import os
import sys

import numpy as np

# this file lives in tests/, so put the repository root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape2sas_core import subunits
from shape2sas_core.models import (
    getPointDistribution,
    rotate_and_translate,
    undo_rotate_and_translate,
)

# one representative set of dimensions per subunit
CASES = {
    "Sphere":             [50.],
    "Cube":               [50.],
    "Cuboid":             [50., 40., 30.],
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
        except SystemExit:
            continue
        raise AssertionError("%s accepted too few dimensions" % name)


def test_rotation_round_trip():
    """undo_rotate_and_translate() must invert rotate_and_translate()

    NOTE: this is what breaks if the inverse rotation is done by negating the
    three Euler angles - that is only correct when at most one is non-zero.
    """
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
        distribution = getPointDistribution(
            ["ellipticalcylinder", "ellipticalcylinder"], [1.0, 1.0],
            [[40., 15., 90.], [40., 15., 90.]],
            [[0, 0, 0], [0, 0, 0]], [list(rotation), list(rotation)],
            True, 8000)
        survivors = len(distribution.x[1])
        assert survivors == 0, ("rotation %s: %d points of the buried subunit "
                                "were not excluded" % (rotation, survivors))


def test_structure_factor_with_several_subunits():
    """the decoupling approximation must cope with subunits of different sizes

    The coordinates are stored per subunit, so the subunits generally hold
    different numbers of points; averaging over that ragged list directly
    raises a ValueError in numpy.
    """
    from shape2sas_core.structure_factors.structure_factors_helpfunctions import calc_A00

    np.random.seed(3)
    distribution = getPointDistribution(
        ["sphere", "sphere"], [1.0, 1.0], [[50.], [30.]],
        [[0, 0, 0], [60, 0, 0]], [[0, 0, 0], [0, 0, 0]], True, 2000)
    assert len(distribution.x[0]) != len(distribution.x[1]), \
        "test needs subunits with different point counts"
    A00 = calc_A00(np.linspace(0.001, 0.5, 20), distribution)
    assert np.all(np.isfinite(A00))


def test_structure_factor_names_resolve():
    """every name documented in the README must resolve to a class

    An unrecognised name used to fall through to S(q) = 1 without any warning,
    so a typo silently removed the structure factor.
    """
    from shape2sas_core.helpfunctions import getStructureFactorClass

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
    except SystemExit:
        return
    raise AssertionError("an unknown structure factor name was accepted")


def test_structure_factor_parameter_counts():
    """a structure factor given the wrong number of parameters must complain"""
    from shape2sas_core import structure_factors

    for name in ("HardSphere", "Aggregation"):
        cls = getattr(structure_factors, name)
        try:
            cls([1.0])
        except SystemExit:
            continue
        raise AssertionError("%s accepted the wrong number of parameters" % name)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print("PASS  %s" % test.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s\n      %s" % (test.__name__, e))
    print("\n%d/%d tests passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
