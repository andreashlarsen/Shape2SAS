"""Simulated measured SESANS data, with noise."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulatedSesans:
    """
    simulated measured SESANS data, drawn by Sesans.simulate

    delta : spin echo lengths [A]
    G     : simulated G(delta) - G(0), i.e. ln(P)/(t lambda^2), with noise [A^-2 cm^-1]
    sigma : uncertainty of G [A^-2 cm^-1]
    """
    delta: np.ndarray
    G: np.ndarray
    sigma: np.ndarray

    def save(self, path) -> None:
        """save the simulated data in the .ses format, without the point at delta = 0"""
        with open(path, 'w') as f:
            f.write('FileFormatVersion      1.0\n')
            f.write('DataFileTitle          Shape2SAS Simulated SESANS data\n')
            f.write('Sample                 Simulated particle\n')
            f.write('Thickness              1.310000\n')
            f.write('Thickness_unit         mm\n')
            f.write('Theta_zmax             0.10000000000000001\n')
            f.write('Theta_zmax_unit        radians\n')
            f.write('Theta_ymax             0.10000000000000001\n')
            f.write('Theta_ymax_unit        radians\n')
            f.write('SpinEchoLength_unit    A\n')
            f.write('Depolarisation_unit    A-2 cm-1\n')
            f.write('Wavelength_unit        A\n')
            f.write('\n')
            f.write('BEGIN_DATA\n')
            f.write('SpinEchoLength Depolarisation Depolarisation_error Wavelength\n')
            for d, G, sigma in zip(self.delta[1:], self.G[1:], self.sigma[1:]):
                f.write('  %-12.5e %-12.5e %-12.5e 2\n' % (d, G, sigma))
