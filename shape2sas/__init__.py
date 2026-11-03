"""Shape2SAS: simulate small-angle scattering from user-defined shapes.

    import shape2sas as sts

    profile = sts.ModelProfile(subunits=["sphere"], dimensions=[[50.0]])
    points = sts.get_point_distribution(profile, Npoints=3000)
    pdd = sts.PairDistanceDistribution(points)
    scattering = sts.get_theoretical_scattering(pdd, sts.uniform_q(), conc=0.02)
    data = sts.simulate_scattering(scattering.q, scattering.I, exposure=500.0)

Units: lengths in A, q in 1/A, scattering length densities (contrasts) in
1e-6 A^-2, and intensities in absolute units, cm^-1.

The names in __all__ are the public API; the shape2sas command is built on the
same functions. Everything else, including the modules they are defined in, is
internal and may change without notice.
"""

__version__ = "2.8.1"

from . import structure_factors, subunits
from .models import ModelPointDistribution, ModelProfile, get_point_distribution
from .simulated_scattering import SimulatedScattering, simulate_scattering
from .theoretical_scattering import (
    PairDistanceDistribution,
    TheoreticalScattering,
    get_theoretical_scattering,
    uniform_q,
)

__all__ = [
    "ModelPointDistribution",
    "ModelProfile",
    "PairDistanceDistribution",
    "SimulatedScattering",
    "TheoreticalScattering",
    "get_point_distribution",
    "get_theoretical_scattering",
    "simulate_scattering",
    "structure_factors",
    "subunits",
    "uniform_q",
]
