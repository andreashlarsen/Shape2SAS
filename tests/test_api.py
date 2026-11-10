#!/usr/bin/python3
"""
Tests for the Python API, i.e. the names exported by shape2sas/__init__.py.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

# this file lives in tests/, so put the repository root on the path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import shape2sas as sts

Q = sts.uniform_q(0.001, 0.5, 200)


def sphere(radius=50., sld=1.0, Npoints=3000, seed=0):
    np.random.seed(seed)
    # the default path needs no optional dependency, so the tests run without pyausaxs
    particle = sts.create_particle(["sphere", [radius], {"sld": sld}])
    return particle.generate_points(Npoints, use_ausaxs=False)


def coordinates(points):
    return np.array([points.x, points.y, points.z, points.sld])


def test_public_names_exist():
    for name in sts.__all__:
        assert hasattr(sts, name), "%s is listed in __all__ but not defined" % name


def test_import_is_light():
    """importing the package must not pull in plotting or the optional AUSAXS backend"""
    code = "import sys, shape2sas; print(sorted(m for m in ('matplotlib', 'pyausaxs') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", "import shape2sas loaded %s" % out.stdout.strip()


def test_forward_scattering_is_absolute():
    """I(0) = conc * V * sld^2 * 1E-4 cm^-1, so it scales with the volume fraction and the squared contrast"""
    radius, sld, conc = 50., 2.0, 0.05
    I = sphere(radius, sld).scattering(Q, conc=conc)
    expected = conc * 4 / 3 * np.pi * radius**3 * sld**2 * 1E-4
    # q[0] is small but not zero, and V is a Monte Carlo estimate
    assert abs(I[0] - expected) <= 0.01 * expected, "I(0) = %.4e, expected %.4e" % (I[0], expected)


def test_pair_distance_distribution():
    radius = 50.
    histogram = sphere(radius).histogram
    r, pr = histogram.r, histogram.pr
    assert np.amax(pr) == 1.0
    assert len(r) == len(pr)
    assert abs(histogram.Rg - np.sqrt(3 / 5) * radius) <= 0.02 * radius
    assert 1.9 * radius <= histogram.dmax <= 2 * radius


def test_structure_factor():
    """the structure factor multiplies the form factor, and is lowered at small q by hard-sphere repulsion"""
    points = sphere()
    S_eff = points.structure_factor(Q, "Hard Sphere", [0.2, 50.])
    assert S_eff[0] < 1.0
    I = points.scattering(Q, structure_factor="Hard Sphere", structure_factor_parameters=[0.2, 50.])
    assert np.allclose(I, points.scattering(Q) * S_eff)
    assert np.allclose(points.structure_factor(Q), 1.0)


def test_interface_roughness():
    points = sphere()
    smooth, rough = points.scattering(Q), points.scattering(Q, sigma_r=5.)
    assert np.allclose(rough, smooth * np.exp(-(Q * 5.)**2 / 2))


def test_subunit_defaults():
    """a subunit has unit contrast, sits at the origin and is not rotated unless told otherwise"""
    np.random.seed(1)
    particle = sts.create_particle()
    particle.add_subunit("sphere", [30.])
    x, y, z, sld = coordinates(particle.generate_points(2000, use_ausaxs=False))
    assert np.all(sld == 1.0)
    assert np.all(x**2 + y**2 + z**2 <= 30.**2)


def test_subunit_by_class_or_name():
    """a subunit can be given as a class or by any of its names, with the same result"""
    from shape2sas import subunits
    particles = []
    for shape in (subunits.Cylinder, "cylinder", "Rod"):
        np.random.seed(2)
        particle = sts.create_particle()
        particle.add_subunit(shape, [20., 60.])
        particles.append(coordinates(particle.generate_points(2000, use_ausaxs=False)))
    for points in particles[1:]:
        assert np.array_equal(points, particles[0])


def test_subunit_as_instance():
    """a subunit created with named dimensions gives the same particle as one given by name and a dimensions list"""
    particles = []
    for args in ((sts.subunits.Cylinder(radius=20., length=60.),), ("cylinder", [20., 60.])):
        np.random.seed(2)
        particle = sts.create_particle()
        particle.add_subunit(*args, com=[0, 0, 10.])
        particles.append(coordinates(particle.generate_points(2000, use_ausaxs=False)))
    assert np.array_equal(particles[0], particles[1])


def test_subunits_given_to_create_particle():
    """create_particle forwards each subunit to add_subunit, in every form add_subunit takes"""
    from shape2sas.subunits import Cylinder, Sphere
    added = sts.create_particle()
    added.add_subunit("sphere", [50.], com=[0, 0, -60.])
    added.add_subunit(Sphere(radius=50.), com=[0, 0, 60.])
    added.add_subunit(Cylinder(radius=10., length=120.))
    added.add_subunit(Cylinder, [10., 20.], sld=2.)
    given = sts.create_particle(
        ["sphere", [50.], {"com": [0, 0, -60.]}],
        [Sphere(radius=50.), {"com": [0, 0, 60.]}],
        Cylinder(radius=10., length=120.),
        (Cylinder, [10., 20.], {"sld": 2.}),
    )
    particles = []
    for particle in (added, given):
        np.random.seed(4)
        particles.append(coordinates(particle.generate_points(3000, use_ausaxs=False)))
    assert np.array_equal(particles[0], particles[1])

    for wrong in ("sphere", [], 50., ["sphere"], ["sphere", [50.], {"color": "red"}]):
        with pytest.raises((ValueError, TypeError)):
            sts.create_particle(wrong)


def test_point_distribution_does_not_change():
    """a point distribution keeps its points and histogram, and is not affected by subunits added to the particle later"""
    np.random.seed(0)
    particle = sts.create_particle()
    particle.add_subunit("sphere", [50.])
    points = particle.generate_points(3000, use_ausaxs=False)
    before = coordinates(points)
    assert points.histogram is points.histogram, "the histogram must not be calculated again"
    with pytest.raises(ValueError):
        points.x[0] = 0.

    particle.add_subunit("sphere", [50.], com=[0, 0, 150.])
    assert np.array_equal(coordinates(points), before)
    dumbbell = particle.generate_points(3000, use_ausaxs=False)
    assert dumbbell.histogram.dmax > 2 * points.histogram.dmax
    I, I_dumbbell = points.scattering(Q), dumbbell.scattering(Q)
    assert not np.allclose(I_dumbbell / I_dumbbell[0], I / I[0], rtol=0.05)


def test_invalid_input_raises_value_error():
    particle = sts.create_particle()
    with pytest.raises(ValueError):
        particle.generate_points(1000)
    with pytest.raises(ValueError):
        particle.add_subunit("no such subunit", [10.])
    with pytest.raises(ValueError):
        particle.add_subunit(dict, [10.])
    with pytest.raises(ValueError):
        particle.add_subunit("cylinder", [10.])
    with pytest.raises(ValueError):
        particle.add_subunit("cylinder")
    with pytest.raises(ValueError):
        particle.add_subunit(sts.subunits.Sphere(radius=10.), [10.])
    with pytest.raises(ValueError):
        particle.add_subunit("sphere", [10.], com=[0, 0])
    with pytest.raises(ValueError):
        sphere().scattering(Q, structure_factor="Hard Sphere", structure_factor_parameters=[0.2])


def test_simulated_scattering():
    """the simulated data scatter around the theoretical intensity, with less noise for longer exposures"""
    I = sphere().scattering(Q)
    np.random.seed(2)
    I_short, sigma_short = sts.simulate_scattering(Q, I, exposure=100.)
    I_long, sigma_long = sts.simulate_scattering(Q, I, exposure=400.)
    assert np.allclose(sigma_short / sigma_long, 2.0), "sigma must scale as 1/sqrt(exposure)"
    pulls = (I_short - I) / sigma_short
    assert abs(np.mean(pulls)) < 0.3 and 0.8 < np.std(pulls) < 1.2
