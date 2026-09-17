#!/usr/bin/python3
"""
Agreement tests between the optional AUSAXS backend (pyausaxs) and the default pair-distance calculation.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pytest

# this file lives in tests/, so put the repository root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape2sas.models import getPointDistribution
from shape2sas.theoretical_scattering import calc_Pq_func, calc_pr_func

Q = np.linspace(0.001, 0.5, 400)


def ausaxs_bin_width():
    """
    the backend's own bin width
    """
    import pyausaxs
    return float(pyausaxs.settings.get("bin_width"))


def quiet(func, *args, **kwargs):
    """run func without its progress chatter reaching the captured output"""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def sphere(Npoints=4000, radius=40., seed=0):
    """one reproducible point distribution, shared by both backends"""
    np.random.seed(seed)
    return quiet(getPointDistribution, ["Sphere"], [1.0], [[radius]],
                 [[0, 0, 0]], [[0, 0, 0]], True, Npoints, None)


def pr(point_distribution, prpoints=100, polydispersity=0.0, use_ausaxs=False):
    return quiet(calc_pr_func, point_distribution, prpoints=prpoints,
                 polydispersity=polydispersity, use_ausaxs=use_ausaxs)


def Pq(point_distribution, **kwargs):
    r, p, _, _ = pr(point_distribution, **kwargs)
    return calc_Pq_func(Q, r, p, 0.001, point_distribution.volume_total)[1]


def max_rel_deviation(a, b):
    """
    largest relative deviation between two form factors
    """
    significant = np.abs(b) > 1e-3
    return np.max(np.abs(a[significant] - b[significant]) / np.abs(b[significant]))


def Rg(r, pr_norm):
    return np.sqrt(abs(np.sum(pr_norm * r**2) / np.sum(pr_norm)) / 2)


# ---------------------------------------------------------------------------
# the distance histogram itself
# ---------------------------------------------------------------------------

def test_histogram_is_the_unordered_pair_sum():
    """
    AUSAXS sums the contrast product over every ordered pair including the
    self-terms; shape2sas wants the sum over unordered pairs i < j only.
    distance_histogram() applies that correction, so its counts must add up
    to (sum(w)^2 - sum(w^2)) / 2 exactly.
    """
    pytest.importorskip("pyausaxs")
    from shape2sas.ausaxs_debye import distance_histogram

    for N in (50, 200, 1000):
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            x, y, z = (rng.uniform(-30, 30, N) for _ in range(3))
            # mixed signs: a core-shell model has negative contrast points
            w = rng.uniform(-1, 2, N)

            _, h = distance_histogram(x, y, z, w)
            expected = (w.sum()**2 - np.sum(w**2)) / 2
            assert abs(h.sum() - expected) <= 1e-6 * abs(expected), (
                "N=%d seed=%d: histogram sums to %.6f, expected %.6f - the "
                "self-term subtraction or the factor of two is wrong"
                % (N, seed, h.sum(), expected))


def test_histogram_grid_is_uniform_from_zero():
    """
    the returned r values must be a regular grid starting at 0, since
    calc_hr_func() hands them straight to the caller as the p(r)
    """
    pytest.importorskip("pyausaxs")
    from shape2sas.ausaxs_debye import distance_histogram

    bin_width = ausaxs_bin_width()
    rng = np.random.default_rng(0)
    x, y, z = (rng.uniform(-30, 30, 500) for _ in range(3))
    r, h = distance_histogram(x, y, z, np.ones(500))

    assert r[0] == 0.0, "first bin is not at r = 0"
    spacing = np.diff(r)
    assert np.allclose(spacing, bin_width), (
        "bin spacing is not uniform at %g A (got %g..%g) - if AUSAXS changed "
        "its default bin width or switched to weighted bins, the p(r) grid "
        "changes with it" % (bin_width, spacing.min(), spacing.max()))
    assert h[-1] != 0, "trailing empty bins were not trimmed"


# ---------------------------------------------------------------------------
# agreement with the default path
# ---------------------------------------------------------------------------

def test_dmax_rg_and_i0_match_default():
    """the moments of p(r) must not depend on which backend produced it"""
    pytest.importorskip("pyausaxs")
    bin_width = ausaxs_bin_width()
    p = sphere()

    r_d, pr_d, prn_d, dmax_d = pr(p, use_ausaxs=False)
    r_a, pr_a, prn_a, dmax_a = pr(p, use_ausaxs=True)

    assert abs(dmax_a - dmax_d) <= 2 * bin_width, (
        "dmax differs by more than a bin: %.4f vs %.4f" % (dmax_a, dmax_d))

    rg_d, rg_a = Rg(r_d, prn_d), Rg(r_a, prn_a)
    assert abs(rg_a - rg_d) <= 0.005 * rg_d, (
        "Rg differs by %.3f%%: %.4f vs %.4f"
        % (100 * abs(rg_a - rg_d) / rg_d, rg_a, rg_d))

    i0_d, i0_a = np.sum(pr_d), np.sum(pr_a)
    assert abs(i0_a - i0_d) <= 0.005 * abs(i0_d), (
        "I(0) differs by %.3f%%: %.6e vs %.6e"
        % (100 * abs(i0_a - i0_d) / abs(i0_d), i0_a, i0_d))


def test_form_factor_matches_default():
    """
    P(q) from the two backends must agree with a converged reference.
    """
    pytest.importorskip("pyausaxs")
    p = sphere()

    reference = Pq(p, prpoints=2000, use_ausaxs=False)
    ausaxs = Pq(p, prpoints=100, use_ausaxs=True)
    default = Pq(p, prpoints=100, use_ausaxs=False)

    err_a = max_rel_deviation(ausaxs, reference)
    err_d = max_rel_deviation(default, reference)

    assert err_a <= 0.02, (
        "AUSAXS P(q) deviates from the converged reference by %.2f%%" % (100 * err_a))
    assert err_a <= err_d, (
        "AUSAXS P(q) (%.2f%%) deviates from the converged reference by %.2f%%" % (100 * err_d))


def test_polydisperse_moments_match_default():
    """polydispersity averaging must give the same moments on either backend"""
    pytest.importorskip("pyausaxs")
    bin_width = ausaxs_bin_width()
    p = sphere()

    r_d, pr_d, prn_d, dmax_d = pr(p, polydispersity=0.1, use_ausaxs=False)
    r_a, pr_a, prn_a, dmax_a = pr(p, polydispersity=0.1, use_ausaxs=True)

    assert np.allclose(np.diff(r_a), bin_width), (
        "the polydisperse AUSAXS path must smear on the native grid")
    assert abs(dmax_a - dmax_d) <= 4 * bin_width, (
        "dmax differs: %.4f vs %.4f" % (dmax_a, dmax_d))

    rg_d, rg_a = Rg(r_d, prn_d), Rg(r_a, prn_a)
    assert abs(rg_a - rg_d) <= 0.005 * rg_d, (
        "Rg differs by %.3f%%" % (100 * abs(rg_a - rg_d) / rg_d))


def test_polydisperse_form_factor_matches_default():
    """
    P(q) under polydispersity must be no worse than the default path.
    """
    pytest.importorskip("pyausaxs")
    p = sphere()

    reference = Pq(p, prpoints=2000, polydispersity=0.05, use_ausaxs=False)
    err_a = max_rel_deviation(Pq(p, prpoints=100, polydispersity=0.05, use_ausaxs=True), reference)
    err_d = max_rel_deviation(Pq(p, prpoints=100, polydispersity=0.05, use_ausaxs=False), reference)

    assert err_a <= 0.03, (
        "AUSAXS P(q) deviates from the converged reference by %.2f%%" % (100 * err_a))
    assert err_a <= err_d, (
        "AUSAXS P(q) (%.2f%%) is worse than the default path (%.2f%%)"
        % (100 * err_a, 100 * err_d))
