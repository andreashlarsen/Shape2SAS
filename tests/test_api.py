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

from shape2sas.structure_factors import Aggregation, HardSphere
from shape2sas.subunits import Cylinder, Sphere

Q = sts.uniform_q(0.001, 0.5, 200)


def sphere(radius=50., sld=1.0, Npoints=3000, seed=0):
    # the default path needs no optional dependency, so the tests run without pyausaxs
    particle = sts.create_particle(Sphere(radius=radius).placed(sld=sld))
    return particle.generate_points(Npoints, rng=seed, use_ausaxs=False)


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
    assert histogram.form_factor(np.array([0.]))[0] == pytest.approx(1.)


def test_structure_factor():
    """the structure factor multiplies the form factor, and is lowered at small q by hard-sphere repulsion"""
    points = sphere()
    hard_spheres = HardSphere(volume_fraction=0.2, radius=50.)
    S_eff = points.effective_structure_factor(Q, hard_spheres)
    assert S_eff[0] < 1.0
    I = points.scattering(Q, structure_factor=hard_spheres)
    assert np.allclose(I, points.scattering(Q) * S_eff)
    assert np.allclose(points.effective_structure_factor(Q), 1.0)


def test_aggregation_sets_the_sesans_range():
    """an aggregate, not the particle, sets the length scale of a SESANS calculation"""
    dmax = 100.
    assert Aggregation(effective_radius=300., particles_per_aggregate=10., fraction_aggregated=0.5).sesans_range(dmax)[1] == 900.
    assert HardSphere(volume_fraction=0.1, radius=50.).sesans_range(dmax)[1] == 3 * dmax


def test_interface_roughness():
    points = sphere()
    smooth, rough = points.scattering(Q), points.scattering(Q, sigma_r=5.)
    assert np.allclose(rough, smooth * np.exp(-(Q * 5.)**2 / 2))


def test_subunit_defaults():
    """a subunit has unit contrast, sits at the origin and is not rotated unless told otherwise"""
    particle = sts.create_particle(Sphere(radius=30.))
    x, y, z, sld = coordinates(particle.generate_points(2000, rng=1, use_ausaxs=False))
    assert np.all(sld == 1.0)
    assert np.all(x**2 + y**2 + z**2 <= 30.**2)
    placement = Sphere(radius=30.).placed()
    assert (placement.sld, placement.com, placement.rotation, placement.rotation_point) == (1., (0., 0., 0.), (0., 0., 0.), (0., 0., 0.))


def test_seed_makes_the_points_reproducible():
    """the same seed, or a generator seeded alike, gives the same points; another seed other points"""
    particle = sts.create_particle(Cylinder(radius=20., length=60.))
    same = [coordinates(particle.generate_points(2000, rng=rng, use_ausaxs=False)) for rng in (7, np.random.default_rng(7))]
    assert np.array_equal(same[0], same[1])
    other = coordinates(particle.generate_points(2000, rng=8, use_ausaxs=False))
    assert not np.array_equal(same[0], other)


def test_subunits_given_to_create_particle():
    """create_particle takes subunits and placed subunits, as add_subunit does"""
    added = sts.create_particle()
    added.add_subunit(Sphere(radius=50.).placed(com=[0, 0, -60.]))
    added.add_subunit(Sphere(radius=50.).placed(com=(0, 0, 60.)))
    added.add_subunit(Cylinder(radius=10., length=120.))
    given = sts.create_particle(
        Sphere(radius=50.).placed(com=(0, 0, -60.)),
        Sphere(radius=50.).placed(com=(0, 0, 60.)),
        Cylinder(radius=10., length=120.),
    )
    points = [coordinates(particle.generate_points(3000, rng=4, use_ausaxs=False)) for particle in (added, given)]
    assert np.array_equal(points[0], points[1])

    for wrong in ("sphere", ["sphere", [50.]], 50., Sphere):
        with pytest.raises(ValueError):
            sts.create_particle(wrong)


def test_point_distribution_does_not_change():
    """a point distribution keeps its points and histogram, and is not affected by subunits added to the particle later"""
    particle = sts.create_particle(Sphere(radius=50.))
    points = particle.generate_points(3000, rng=0, use_ausaxs=False)
    before = coordinates(points)
    assert points.histogram is points.histogram, "the histogram must not be calculated again"
    with pytest.raises(ValueError):
        points.x[0] = 0.

    particle.add_subunit(Sphere(radius=50.).placed(com=(0, 0, 150.)))
    assert np.array_equal(coordinates(points), before)
    dumbbell = particle.generate_points(3000, rng=0, use_ausaxs=False)
    assert dumbbell.histogram.dmax > 2 * points.histogram.dmax
    I, I_dumbbell = points.scattering(Q), dumbbell.scattering(Q)
    assert not np.allclose(I_dumbbell / I_dumbbell[0], I / I[0], rtol=0.05)


def test_invalid_input_raises_value_error():
    particle = sts.create_particle()
    with pytest.raises(ValueError):
        particle.generate_points(1000)
    for wrong in ("sphere", Sphere, 50.):
        with pytest.raises(ValueError):
            particle.add_subunit(wrong)
    with pytest.raises(ValueError):
        Sphere(radius=10.).placed(com=(0, 0))
    with pytest.raises(ValueError):
        sphere().scattering(Q, structure_factor="hardsphere")


def test_placement_keywords_are_checked():
    """a subunit is placed with placed(), whose keywords are checked; add_subunit takes the subunit and nothing else"""
    particle = sts.create_particle()
    with pytest.raises(TypeError):
        Sphere(radius=10.).placed(colour="red")
    with pytest.raises(TypeError):
        particle.add_subunit(Sphere(radius=10.), sld=2.)
    with pytest.raises(TypeError):
        particle.add_subunit(Sphere(radius=10.), [10.])


def test_simulated_scattering():
    """the simulated data scatter around the theoretical intensity, with less noise for longer exposures"""
    I = sphere().scattering(Q)
    I_short, sigma_short = sts.simulate_scattering(Q, I, exposure=100., rng=2)
    I_long, sigma_long = sts.simulate_scattering(Q, I, exposure=400., rng=2)
    assert np.allclose(sigma_short / sigma_long, 2.0), "sigma must scale as 1/sqrt(exposure)"
    pulls = (I_short - I) / sigma_short
    assert abs(np.mean(pulls)) < 0.3 and 0.8 < np.std(pulls) < 1.2
    assert np.array_equal(sts.simulate_scattering(Q, I, exposure=100., rng=2)[0], I_short), "the same seed must give the same data"
