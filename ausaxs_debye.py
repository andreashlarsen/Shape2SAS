"""
Optional AUSAXS backend for the Debye scattering calculation.
"""

import numpy as np

_configured = False


def _configure():
    """Apply AUSAXS histogram settings once per process."""
    global _configured
    if _configured:
        return
    import pyausaxs as ausaxs
    # Shape2SAS point clouds are randomly distributed (not on a lattice/bonded
    # structure), so unweighted histogram bins are both faster and appropriate.
    ausaxs.settings.histogram(weighted_bins=False)
    _configured = True


def distance_histogram(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pairwise-distance histogram of a set of weighted points via pyAUSAXS.

    input:
    x, y, z : point coordinates
    w : weight (contrast) associated with each point

    output:
    r : bin distances (regular grid, AUSAXS's native bin width), trimmed to the last nonzero bin
    h : histogram, weighted by the contrast product of each pair

    raises ImportError if pyausaxs is not installed, or RuntimeError if the
    AUSAXS backend fails to evaluate the calculation.
    """
    import pyausaxs as ausaxs
    _configure()
    mol = ausaxs.create_molecule(np.asarray(x), np.asarray(y), np.asarray(z), np.asarray(w))
    hist = mol.distance_histogram()
    bins = hist.bins()
    counts = np.array(hist.counts_aa(), dtype=np.float64, copy=True)

    # AUSAXS's histogram sums the contrast product over every ordered pair
    # (i, j), including self-terms (i == j) landing in the r=0 bin.
    counts[0] -= np.sum(np.asarray(w, dtype=np.float64)**2)
    counts /= 2.0

    nonzero = np.nonzero(counts)[0]
    last = nonzero[-1] if len(nonzero) else 0
    return bins[:last + 1], counts[:last + 1]