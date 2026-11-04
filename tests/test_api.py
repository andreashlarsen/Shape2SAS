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


def sphere(radius=50., sld=1.0, Npoints=3000):
    return sts.get_point_distribution(sts.ModelProfile(dimensions=[[radius]], sld=[sld]), Npoints)


def pair_distances(point_distribution):
    # the default path needs no optional dependency, so the tests run without pyausaxs
    return sts.PairDistanceDistribution(point_distribution, use_ausaxs=False)


def scattering(point_distribution, **kwargs):
    return sts.get_theoretical_scattering(pair_distances(point_distribution), Q, **kwargs)


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
    result = scattering(sphere(radius, sld), conc=conc)
    expected = conc * 4 / 3 * np.pi * radius**3 * sld**2 * 1E-4
    # q[0] is small but not zero, and V is a Monte Carlo estimate
    assert abs(result.I[0] - expected) <= 0.01 * expected, "I(0) = %.4e, expected %.4e" % (result.I[0], expected)
    assert np.allclose(result.I, conc * result.Pq)


def test_pair_distance_distribution():
    radius = 50.
    result = pair_distances(sphere(radius))
    assert np.amax(result.pr) == 1.0
    assert len(result.r) == len(result.pr)
    assert abs(result.Rg - np.sqrt(3 / 5) * radius) <= 0.02 * radius
    assert 1.9 * radius <= result.dmax <= 2 * radius


def test_structure_factor():
    """the structure factor multiplies the form factor, and is lowered at small q by hard-sphere repulsion"""
    points = sphere()
    result = scattering(points, structure_factor="Hard Sphere", structure_factor_parameters=[0.2, 50.])
    assert result.S_eff[0] < 1.0
    assert np.allclose(result.I, 0.02 * result.Pq * result.S_eff)

    no_structure = scattering(points)
    assert np.allclose(no_structure.S_eff, 1.0)
    assert np.allclose(no_structure.Pq, result.Pq), "the structure factor must not change the form factor"


def test_interface_roughness():
    points = sphere()
    smooth, rough = scattering(points), scattering(points, sigma_r=5.)
    expected = smooth.I * np.exp(-(smooth.q * 5.)**2 / 2)
    assert np.allclose(rough.I, expected)


def test_defaults_per_subunit():
    """sld, com, rotation and rotation points default to one value per subunit"""
    profile = sts.ModelProfile(subunits=["sphere", "cylinder"], dimensions=[[30.], [10., 80.]])
    points = sts.get_point_distribution(profile, 2000)
    assert np.all(points.sld == 1.0)


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError):
        sts.get_point_distribution(sts.ModelProfile(subunits=["no such subunit"]))
    with pytest.raises(ValueError):
        sts.get_point_distribution(sts.ModelProfile(subunits=["cylinder"], dimensions=[[10.]]))
    with pytest.raises(ValueError):
        sts.get_point_distribution(sts.ModelProfile(subunits=["sphere"], sld=[1.0, 2.0]))
    with pytest.raises(ValueError):
        scattering(sphere(), structure_factor="Hard Sphere", structure_factor_parameters=[0.2])


def test_simulated_scattering():
    """the simulated data scatter around the theoretical intensity, with less noise for longer exposures"""
    theory = scattering(sphere())
    np.random.seed(2)
    short = sts.simulate_scattering(theory.q, theory.I, exposure=100.)
    long = sts.simulate_scattering(theory.q, theory.I, exposure=400.)
    assert np.array_equal(short.q, theory.q)
    assert np.allclose(short.sigma / long.sigma, 2.0), "sigma must scale as 1/sqrt(exposure)"
    pulls = (short.I - theory.I) / short.sigma
    assert abs(np.mean(pulls)) < 0.3 and 0.8 < np.std(pulls) < 1.2
