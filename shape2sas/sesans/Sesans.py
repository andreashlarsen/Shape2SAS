"""The SESANS projected correlation function G(delta) of a total intensity."""

from dataclasses import dataclass

import numpy as np

from ..theoretical_scattering import write_columns
from .sesans_calculations import simulate_sesans
from .SimulatedSesans import SimulatedSesans


@dataclass(frozen=True)
class Sesans:
    """
    projected correlation function, calculated by TotalIntensity.sesans

    delta : spin echo lengths [A]
    G     : projected correlation function G(delta), in absolute units [A^-2 cm^-1]
    """
    delta: np.ndarray
    G: np.ndarray

    def simulate(self, error: float = 0.02, *, rng: np.random.Generator | int | None = None) -> SimulatedSesans:
        """
        simulate measured SESANS data, with noise

        error : noise relative to the depth of G(delta) - G(0)
        rng   : random number generator (np.random.Generator), or a seed for one; None draws a fresh seed
        """
        G, sigma = simulate_sesans(self.delta, self.G, error, rng=rng)
        return SimulatedSesans(self.delta, G, sigma)

    def save(self, path) -> None:
        """save G(delta) as a text file with the columns delta, G"""
        write_columns(path, ('delta', 'G'), (self.delta, self.G), title='Theoretical SESANS data')
