#!/usr/bin/python3
"""
Validate the subunits against the analytical form factors in sasmodels, the model library behind SasView.

Every subunit with a sasmodels counterpart must reproduce its P(q) and its volume. The torus and the hyperboloid have no counterpart,
and are only covered by the self-consistency tests in test_subunits.py.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sasmodels")
from sasmodels.core import load_model
from sasmodels.direct_model import call_Fq, call_kernel

from shape2sas import subunits
from shape2sas.particle import create_particle
from shape2sas.theoretical_scattering import PairDistanceDistribution

# subunit, its dimensions, and the same body in sasmodels
CASES = [
    pytest.param("Sphere", [50.], "sphere", dict(radius=50.), id="Sphere"),
    pytest.param("Cube", [50.], "parallelepiped", dict(length_a=50., length_b=50., length_c=50.), id="Cube"),
    pytest.param("Cuboid", [50., 40., 30.], "parallelepiped", dict(length_a=50., length_b=40., length_c=30.), id="Cuboid"),
    pytest.param("Octahedron", [35., 35., 35.], "truncated_octahedron", dict(radius_a=35., truncation=0.), id="Octahedron-regular"),
    pytest.param("Octahedron", [40., 30., 25.], "truncated_octahedron",
                 dict(radius_a=40., b2a_ratio=30 / 40, c2a_ratio=25 / 40, truncation=0.), id="Octahedron-general"),
    pytest.param("TruncatedOctahedron", [40., 30., 25., 0.3], "truncated_octahedron",
                 dict(radius_a=40., b2a_ratio=30 / 40, c2a_ratio=25 / 40, truncation=0.3), id="TruncatedOctahedron-general"),
    pytest.param("TruncatedOctahedron", [35., 35., 35., 0.5], "truncated_octahedron",
                 dict(radius_a=35., truncation=0.5), id="TruncatedOctahedron-cuboctahedron"),
    pytest.param("Cylinder", [30., 80.], "cylinder", dict(radius=30., length=80.), id="Cylinder"),
    pytest.param("CircularDisc", [30., 20.], "cylinder", dict(radius=30., length=20.), id="CircularDisc"),
    # sasmodels takes the inner radius and the wall thickness
    pytest.param("CylinderRing", [50., 30., 40.], "hollow_cylinder", dict(radius=30., thickness=20., length=40.), id="CylinderRing"),
    pytest.param("Ellipsoid", [50., 30., 20.], "triaxial_ellipsoid",
                 dict(radius_equat_minor=50., radius_equat_major=30., radius_polar=20.), id="Ellipsoid"),
    pytest.param("EllipticalCylinder", [40., 20., 60.], "elliptical_cylinder",
                 dict(radius_minor=20., axis_ratio=2., length=60.), id="EllipticalCylinder"),
    pytest.param("Disc", [40., 20., 15.], "elliptical_cylinder", dict(radius_minor=20., axis_ratio=2., length=15.), id="Disc"),
    # outer side 50 and inner side 30 leave walls of thickness 10
    pytest.param("HollowCube", [50., 30.], "hollow_rectangular_prism", dict(length_a=50., thickness=10.), id="HollowCube"),
    # a core-shell sphere whose core matches the solvent
    pytest.param("HollowSphere", [50., 30.], "core_shell_sphere", dict(radius=30., thickness=20.), id="HollowSphere"),
    # with eps = 1 and t = s = 2p the superellipsoid is a superball of edge 2R
    pytest.param("Superellipsoid", [50., 1., 4., 4.], "superball", dict(length_a=100., exponent_p=2.), id="Superellipsoid"),
    # sasmodels only has spheroidal shells (a = b); the shell is centred on the given semi-axes, so the core is T/2 smaller and the shell T thick
    pytest.param("Ellipsoid_shell", [40., 40., 60., 5.], "core_shell_ellipsoid",
                 dict(radius_equat_core=37.5, x_core=57.5 / 37.5, thick_shell=5., x_polar_shell=1.), id="Ellipsoid_shell"),
]

Q = np.linspace(0.001, 0.5, 200)
Q0 = np.array([1e-5])


def sasmodels_reference(name, pars):
    """
    P(q), normalised to 1 at q = 0, and the volume of the scattering material

    The body is given unit contrast; a core, where the model has one, is matched to the solvent. sasmodels returns the amplitude F in
    units where 100 F(0) is the contrast times the volume, so 100 F(0) is the volume.
    """
    model = load_model(name)
    names = {p.name for p in model.info.parameters.kernel_parameters}
    contrast = dict(sld=1.) if "sld" in names else dict(sld_shell=1., sld_core=0.)
    pars = dict(sld_solvent=0., scale=1., background=0., **contrast, **pars)

    I0 = call_kernel(model.make_kernel([Q0]), pars)[0]
    Pq = call_kernel(model.make_kernel([Q]), pars) / I0
    F1 = call_Fq(model.make_kernel([Q0]), pars)[0]
    return Pq, 100 * F1[0]


@pytest.mark.parametrize("subunit, dimensions, model, pars", CASES)
def test_volume_matches_sasmodels(subunit, dimensions, model, pars):
    """getVolume() must give the volume of the sasmodels body"""
    expected = sasmodels_reference(model, pars)[1]
    volume = getattr(subunits, subunit)(*dimensions).getVolume()
    assert volume == pytest.approx(expected, rel=1e-3), (
        "%s %s has volume %.6g, sasmodels %s gives %.6g" % (subunit, dimensions, volume, model, expected))


@pytest.mark.parametrize("subunit, dimensions, model, pars", CASES)
def test_form_factor_matches_sasmodels(subunit, dimensions, model, pars):
    """
    the simulated P(q) must match the analytical one in sasmodels

    Near the minima the sampling noise in P(q) scales as 1/sqrt(N P), so P(q) is averaged over a few independent point clouds and
    compared by its mean relative deviation.
    """
    expected = sasmodels_reference(model, pars)[0]

    simulated = []
    for seed in range(4):
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            particle = create_particle(Npoints=10000)
            particle.add_subunit(subunit, list(dimensions))
            distribution = particle._points()
            pr = PairDistanceDistribution(distribution, prpoints=100, use_ausaxs=False)
        simulated.append(pr.calc_Pq(Q))
    simulated = np.mean(simulated, axis=0)

    significant = expected > 1e-3
    deviation = np.mean(np.abs(simulated[significant] - expected[significant]) / expected[significant])
    assert deviation < 0.04, (
        "simulated P(q) of %s %s deviates from sasmodels %s by %.1f%% on average" % (subunit, dimensions, model, 100 * deviation))
