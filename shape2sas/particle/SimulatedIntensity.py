"""Simulated measured data, with noise, from a total intensity."""

from dataclasses import dataclass

import numpy as np

from ..theoretical_scattering import write_columns


@dataclass(frozen=True)
class SimulatedIntensity:
    """
    simulated measured data, drawn by TotalIntensity.simulate

    q     : q values [1/A]
    I     : simulated intensity, with noise [cm^-1]
    sigma : uncertainty of I [cm^-1]
    """
    q: np.ndarray
    I: np.ndarray
    sigma: np.ndarray

    def save(self, path) -> None:
        """save the simulated data as a text file with the columns q, I, sigma"""
        write_columns(path, ('q', 'I', 'sigma'), (self.q, self.I, self.sigma),
                      title='Simulated SAXS data with noise\n'
                            'sigma generated using Sedlak et al, k=100000, c=0.55, https://doi.org/10.1107/S1600576717003077, and rebinned with 10 per bin)')
