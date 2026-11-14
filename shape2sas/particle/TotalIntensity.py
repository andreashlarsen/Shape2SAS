"""The total scattering of particles: their intensity and the structure factor it includes."""

from dataclasses import dataclass

import numpy as np

from ..sesans import Sesans, calc_G_sesans
from ..simulated_scattering import simulate_scattering
from ..theoretical_scattering import write_columns
from .SimulatedIntensity import SimulatedIntensity


@dataclass(frozen=True)
class TotalIntensity:
    """
    total scattering calculated by PointDistribution.scattering: the intensity of the single particles times their structure factor

    q : q values [1/A]
    I : total intensity in absolute units [cm^-1], including S
    S : effective structure factor S_eff(q)
    """
    q: np.ndarray
    I: np.ndarray
    S: np.ndarray

    def simulate(self, exposure: float = 500.0, *, rng: np.random.Generator | int | None = None) -> SimulatedIntensity:
        """
        simulate measured data, with noise following Sedlak et al. 2017 (https://doi.org/10.1107/S1600576717003077)

        exposure : exposure (in arbitrary units); a longer exposure gives less noise
        rng      : random number generator (np.random.Generator), or a seed for one; None draws a fresh seed
        """
        I, sigma = simulate_scattering(self.q, self.I, exposure, rng=rng)
        return SimulatedIntensity(self.q, I, sigma)

    def sesans(self, delta) -> Sesans:
        """
        projected correlation function G(delta) for SESANS, the Hankel transform of I(q).
        q must be equidistant and reach far enough for the transform, see StructureFactor.sesans_range

        delta : spin echo lengths [A]
        """
        delta = np.asarray(delta, dtype=float)
        return Sesans(delta, calc_G_sesans(self.q, delta, self.I))

    def save_intensity(self, path) -> None:
        """save the total intensity as a text file with the columns q, I"""
        write_columns(path, ('q', 'I'), (self.q, self.I), title='Theoretical SAS data')

    def save_structure_factor(self, path) -> None:
        """save the effective structure factor as a text file with the columns q, S"""
        write_columns(path, ('q', 'S'), (self.q, self.S), title='Structure factor SAS data')
