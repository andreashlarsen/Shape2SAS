"""The total scattering of particles: their intensity and the structure factor it includes."""

from dataclasses import dataclass

import numpy as np

from ..theoretical_scattering import write_columns


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

    def save_intensity(self, path) -> None:
        """save the total intensity as a text file with the columns q, I"""
        write_columns(path, ('q', 'I'), (self.q, self.I), title='Theoretical SAS data')

    def save_structure_factor(self, path) -> None:
        """save the effective structure factor as a text file with the columns q, S"""
        write_columns(path, ('q', 'S'), (self.q, self.S), title='Structure factor SAS data')
