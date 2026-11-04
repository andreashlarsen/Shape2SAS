#!/usr/bin/python3
"""
Self-consistency tests for the subunits, the structure factors and the
rotation machinery.
"""

import os
import sys

import numpy as np
import pytest

# this file lives in tests/, so put the repository root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape2sas import subunits
from shape2sas.models import ModelProfile, get_point_distribution
from shape2sas.subunits.subunits_helpfunctions import (
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

N = 200000
ROTATIONS = [(0, 0, 0), (30, 0, 0), (0, 40, 0), (0, 0, 50), (30, 40, 50), (11, -73, 145)]


def measure_body(obj, rng):
    """
    sample a box larger than the one the subunit claims to fit in, and return
    the volume of the body that checkOverlap() describes, and the number of
    points inside the body but outside getHalfWidths()
    """
    half = np.asarray(obj.getHalfWidths(), dtype=float)
    x, y, z = rng.uniform(-1.5 * half, 1.5 * half, (N, 3)).T
    inside = np.ones(N, dtype=bool)
    inside[obj.checkOverlap(x, y, z)] = False
    beyond = inside & np.any(np.abs(np.vstack([x, y, z]).T) > half, axis=1)
    return np.count_nonzero(inside) / N * np.prod(3 * half), np.count_nonzero(beyond)


def test_subunits_are_self_consistent():
    """getHalfWidths() and getVolume() must agree with the body that checkOverlap() describes"""
    rng = np.random.default_rng(0)
    failures = []
    for name, dimensions in CASES.items():
        obj = getattr(subunits, name)(list(dimensions))
        volume, beyond = measure_body(obj, rng)

        # the cloud extends beyond the box, so the body must lie within it
        if beyond:
            failures.append("%s: %d points outside getHalfWidths() are inside the body"
                            % (name, beyond))

        # getVolume() sets the point density, so it must match the body
        ratio = volume / obj.getVolume()
        if not 0.95 < ratio < 1.05:
            failures.append("%s: the body has volume %.4g, but getVolume() gives %.4g (ratio %.3f)"
                            % (name, volume, obj.getVolume(), ratio))
    assert not failures, "\n".join(failures)


@pytest.mark.xfail(strict=True, reason="Torus.getVolume() is 2 pi^2 R r^2, which only holds for r <= R")
def test_self_intersecting_torus_volume():
    """
    with r > R a torus intersects itself, and its volume is no longer 2 pi^2 R r^2

    tools/run_all_subunits.sh uses R = 5, r = 10, whose body is 8% larger than
    getVolume(): the model gets 8% more points than requested, and its
    volume_total, and so I(0), is 8% too low
    """
    torus = subunits.Torus([5., 10.])
    volume, _ = measure_body(torus, np.random.default_rng(0))
    assert 0.95 < volume / torus.getVolume() < 1.05


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
    """a subunit placed inside an identical one must add no points

    Twice the points for twice the volume is the same density, so both models
    are cut out of the same cloud. An asymmetric subunit is used, because a
    symmetric one hides errors in the inverse rotation.
    """
    for rotation in ROTATIONS:
        single = get_point_distribution(ModelProfile(
            subunits=["ellipticalcylinder"], dimensions=[[40., 15., 90.]],
            rotation=[list(rotation)]), 4000)
        double = get_point_distribution(ModelProfile(
            subunits=["ellipticalcylinder", "ellipticalcylinder"],
            dimensions=[[40., 15., 90.], [40., 15., 90.]],
            rotation=[list(rotation), list(rotation)]), 8000)
        extra = len(double.x) - len(single.x)
        assert extra == 0, ("rotation %s: the buried subunit added %d points"
                            % (rotation, extra))


def test_structure_factor_with_several_subunits():
    """the decoupling approximation must cope with subunits of different sizes"""
    from shape2sas.structure_factors.structure_factors_helpfunctions import calc_A00

    distribution = get_point_distribution(ModelProfile(
        subunits=["sphere", "sphere"], dimensions=[[50.], [30.]],
        com=[[0, 0, 0], [60, 0, 0]]), 2000)
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