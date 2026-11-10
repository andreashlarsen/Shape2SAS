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


def test_wrong_number_of_dimensions_is_rejected():
    """a dimensions list of the wrong length must give an InputError (a ValueError) that names the parameters"""
    for name, dimensions in CASES.items():
        cls = getattr(subunits, name)
        for wrong in (list(dimensions)[:-1], list(dimensions) + [1.]):
            with pytest.raises(ValueError, match=", ".join(cls.parameters())):
                cls.from_dimensions(wrong)


def test_named_and_listed_dimensions_give_the_same_subunit():
    """keywords, positional arguments and from_dimensions() are three spellings of the same subunit"""
    for name, dimensions in CASES.items():
        cls = getattr(subunits, name)
        keywords = dict(zip(cls.parameters(), dimensions))
        for obj in (cls(*dimensions), cls.from_dimensions(dimensions)):
            assert obj.volume() == cls(**keywords).volume(), name


@pytest.mark.parametrize("package", [subunits, structure_factors], ids=lambda package: package.__name__)
def test_every_class_is_exported(package):
    """every class in the package must be imported and listed in its __init__.py, or users cannot select it"""
    defined = set()
    for module in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(package.__name__ + "." + module.name)
        defined |= {name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__}
    assert defined == set(package.__all__), "not exported in %s/__init__.py: %s" % (package.__name__, sorted(defined - set(package.__all__)))


def test_every_subunit_derives_from_subunit():
    """all subunit classes share the Subunit interface, and CASES covers all of them"""
    names = {name for name in subunits.__all__
             if issubclass(getattr(subunits, name), subunits.Subunit) and name != "Subunit"}
    assert names == set(CASES), "subunits missing from CASES: %s" % sorted(names - set(CASES))
    for name, dimensions in CASES.items():
        cls = getattr(subunits, name)
        assert isinstance(cls(*dimensions), subunits.Subunit), name
        assert len(dimensions) == len(cls.parameters()), name


def test_malformed_subunits_are_rejected():
    """a subunit must define its own aliases, and implement every method"""
    with pytest.raises(TypeError):
        class NoAliases(subunits.Subunit):
            pass
    with pytest.raises(TypeError):
        # inheriting the aliases would give two subunits the same names
        class InheritedAliases(subunits.Sphere):
            pass

    class NoMethods(subunits.Subunit):
        aliases = ["nomethods"]
    with pytest.raises(TypeError):
        NoMethods()


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
    """a parameter list of the wrong length must give an InputError (a ValueError) that names the parameters"""
    for cls in (structure_factors.HardSphere, structure_factors.Aggregation, structure_factors.NoStructure):
        n = len(cls.parameters())
        for wrong in ([1.0] * (n + 1), [1.0] * (n - 1)) if n else ([1.0],):
            with pytest.raises(ValueError, match=", ".join(cls.parameters())):
                cls.from_parameters(wrong)
        assert isinstance(cls.from_parameters([1.0] * n), structure_factors.StructureFactor)


def test_malformed_structure_factors_are_rejected():
    """a structure factor must define its own aliases, and implement structure_factor()"""
    with pytest.raises(TypeError):
        class NoAliases(structure_factors.StructureFactor):
            def structure_factor(self, q):
                return q
    class NoMethods(structure_factors.StructureFactor):
        aliases = ["nomethods"]
    with pytest.raises(TypeError):
        NoMethods()
