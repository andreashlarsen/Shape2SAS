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
    particle = sts.create_particle(Npoints=Npoints, use_ausaxs=False)
    particle.add_subunit("sphere", [radius], sld=sld)
    return particle


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
    particle = sphere(radius)
    r, pr = particle.pr()
    assert np.amax(pr) == 1.0
    assert len(r) == len(pr)
    assert abs(particle.Rg - np.sqrt(3 / 5) * radius) <= 0.02 * radius
    assert 1.9 * radius <= particle.dmax <= 2 * radius


def test_structure_factor():
    """the structure factor multiplies the form factor, and is lowered at small q by hard-sphere repulsion"""
    particle = sphere()
    S_eff = particle.structure_factor(Q, "Hard Sphere", [0.2, 50.])
    assert S_eff[0] < 1.0
    I = particle.scattering(Q, structure_factor="Hard Sphere", structure_factor_parameters=[0.2, 50.])
    assert np.allclose(I, particle.scattering(Q) * S_eff)
    assert np.allclose(particle.structure_factor(Q), 1.0)


def test_interface_roughness():
    particle = sphere()
    smooth, rough = particle.scattering(Q), particle.scattering(Q, sigma_r=5.)
    assert np.allclose(rough, smooth * np.exp(-(Q * 5.)**2 / 2))


def test_subunit_defaults():
    """a subunit has unit contrast, sits at the origin and is not rotated unless told otherwise"""
    np.random.seed(1)
    particle = sts.create_particle(Npoints=2000, use_ausaxs=False)
    particle.add_subunit("sphere", [30.])
    x, y, z, sld = particle.points()
    assert np.all(sld == 1.0)
    assert np.all(x**2 + y**2 + z**2 <= 30.**2)


def test_subunit_by_class_or_name():
    """a subunit can be given as a class or by any of its names, with the same result"""
    from shape2sas import subunits
    particles = []
    for shape in (subunits.Cylinder, "cylinder", "Rod"):
        np.random.seed(2)
        particle = sts.create_particle(Npoints=2000, use_ausaxs=False)
        particle.add_subunit(shape, [20., 60.])
        particles.append(particle.points())
    for points in particles[1:]:
        assert np.array_equal(points, particles[0])


def test_adding_a_subunit_updates_the_particle():
    """the points, p(r) and P(q) are kept between calls, but recalculated once the particle changes"""
    particle = sphere()
    assert np.array_equal(particle.points()[0], particle.points()[0]), "the random points must not be drawn again"
    dmax, I = particle.dmax, particle.scattering(Q)
    particle.add_subunit("sphere", [50.], com=[0, 0, 150.])
    assert particle.dmax > 2 * dmax
    I_dumbbell = particle.scattering(Q)
    assert not np.allclose(I_dumbbell / I_dumbbell[0], I / I[0], rtol=0.05), "the form factor must be recalculated"


def test_invalid_input_raises_value_error():
    particle = sts.create_particle()
    with pytest.raises(ValueError):
        particle.scattering(Q)
    with pytest.raises(ValueError):
        particle.add_subunit("no such subunit", [10.])
    with pytest.raises(ValueError):
        particle.add_subunit(dict, [10.])
    with pytest.raises(ValueError):
        particle.add_subunit("cylinder", [10.])
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
