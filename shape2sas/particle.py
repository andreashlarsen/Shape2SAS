"""A particle built from subunits, the random points filling it, and the scattering they give."""

from dataclasses import dataclass
from functools import cached_property
import os

import numpy as np

from .helpfunctions import InputError
from .models import ModelPointDistribution, GenerateAllPoints
from .structure_factors import NoStructure, StructureFactor
from .subunits.Subunit import Placement, Subunit
from .theoretical_scattering import PairDistanceDistribution, calc_Iq_func, write_columns


class PointDistribution:
    """
    the random points filling a particle, drawn by Particle.generate_points, and the scattering calculated from them

    the points never change: other subunits or other settings need a new point distribution from generate_points.
    the histogram is therefore calculated on first use and kept.
    save, write_pdb and plot_projections write the points to a text file, a PDB file for visualization, or a figure.

    attributes:
    x, y, z        : coordinates of the points [A], read-only
    sld            : excess scattering length density (contrast) of each point [1e-6 A^-2], read-only
    volume         : volume of the particle, without the overlap between subunits if that is excluded [A^3]
    prpoints       : number of bins in p(r)
    polydispersity : relative polydispersity of the particle size
    use_ausaxs     : whether the AUSAXS backend is tried first for the histogram
    """

    def __init__(self, distribution: ModelPointDistribution, prpoints, polydispersity, use_ausaxs):
        self._distribution = distribution
        self.x, self.y, self.z, self.sld = (np.concatenate(a) for a in (distribution.x, distribution.y, distribution.z, distribution.sld))
        for a in (self.x, self.y, self.z, self.sld):
            a.setflags(write=False)
        self.volume = distribution.volume_total
        self.prpoints = prpoints
        self.polydispersity = polydispersity
        self.use_ausaxs = use_ausaxs

    @cached_property
    def histogram(self) -> PairDistanceDistribution:
        """
        the pair distance distribution of the points, with
            r, pr                : p(r), with roughly prpoints bins, normalized so max p(r) = 1
            r_hist, hr           : full-resolution h(r)
            dmax, Rg             : maximum distance and radius of gyration [A]
            form_factor(q)       : form factor, normalized so P(0) = 1
            forward_scattering() : forward scattering per unit volume fraction, V * contrast^2 [cm^-1]
        """
        return PairDistanceDistribution(self._distribution, self.prpoints, self.polydispersity, self.use_ausaxs)

    def scattering(self, q, *, conc: float = 0.02, structure_factor: StructureFactor | None = None, sigma_r: float = 0.0) -> 'TotalIntensity':
        """
        scattering of the particles in absolute units [cm^-1], I(q) = conc * P(q) * S_eff(q) * exp(-(q sigma_r)^2 / 2),
        where the form factor P(q) is per unit volume fraction, so P(0) = V * contrast^2,
        and S_eff(q) is the structure factor corrected for the shape of the particles (decoupling approximation)

        q                : q values [1/A]
        conc             : volume fraction of the particles
        structure_factor : e.g. structure_factors.HardSphere(volume_fraction=0.2, radius=50); None for dilute particles, S = 1
        sigma_r          : interface roughness [A]
        """
        if structure_factor is None:
            structure_factor = NoStructure()
        if not isinstance(structure_factor, StructureFactor):
            raise InputError("a structure factor must be given as e.g. structure_factors.HardSphere(volume_fraction=0.2, radius=50), "
                             "but %r was given" % (structure_factor,))
        q = np.asarray(q, dtype=float)
        # the decoupling approximation needs the form factor normalized so P(0) = 1
        Pq_norm = self.histogram.form_factor(q)
        S_eff = structure_factor.effective_structure_factor(q, self._distribution, Pq_norm)
        I = conc * self.histogram.forward_scattering() * calc_Iq_func(q, Pq_norm, S_eff, sigma_r)
        return TotalIntensity(q, I, S_eff)

    def save(self, path) -> None:
        """save the points as a text file with the columns x, y, z, sld"""
        np.savetxt(path, np.column_stack((self.x, self.y, self.z, self.sld)), fmt='%f', header='x y z sld')

    def write_pdb(self, path, *, title: str | None = None) -> None:
        """
        save the points as a PDB file, for visualization only, e.g. with PyMOL

        each point is a dummy atom, carbon (C) for positive, hydrogen (H) for zero and oxygen (O) for negative contrast;
        only the sign of the contrast is kept, so scattering calculated from the file with e.g. CRYSOL, Pepsi-SAXS or FoXS is wrong

        title : title in the file header, by default the file name without extension
        """
        if title is None:
            title = os.path.splitext(os.path.basename(path))[0]
        atoms = np.where(self.sld > 0, 'C', np.where(self.sld == 0, 'H', 'O'))
        with open(path, 'w') as f:
            f.write('TITLE    POINT SCATTER FOR MODEL: %s\n' % title)
            f.write('REMARK   GENERATED WITH Shape2SAS\n')
            f.write('REMARK   EACH BEAD REPRESENTED BY DUMMY ATOM\n')
            f.write('REMARK   CARBON, C : POSITIVE EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   HYDROGEN, H : ZERO EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   OXYGEN, O : NEGATIVE EXCESS SCATTERING LENGTH\n')
            f.write('REMARK   ACCURATE SCATTERING LENGTH DENSITY INFORMATION NOT INCLUDED\n')
            f.write('REMARK   OBS: WILL NOT GIVE CORRECT RESULTS IF SCATTERING IS CALCULATED FROM THIS MODEL WITH E.G CRYSOL, PEPSI-SAXS, FOXS, CAPP OR THE LIKE!\n')
            f.write('REMARK   ONLY FOR VISUALIZATION, E.G. WITH PYMOL\n')
            f.write('REMARK    \n')
            for i, (atom, xi, yi, zi) in enumerate(zip(atoms, self.x, self.y, self.z)):
                f.write('ATOM  %6i %s   ALA A%6i  %8.3f%8.3f%8.3f  1.00  0.00           %s \n' % (i, atom, i, xi, yi, zi, atom))
            f.write('END')

    @property
    def extent(self) -> float:
        """largest distance of a point from the origin along x, y or z [A]"""
        return float(max(np.amax(np.abs(a)) for a in (self.x, self.y, self.z)))

    def plot_projections(self, path=None, *, color='blue', extent: float | None = None):
        """
        plot the points projected onto the (x,z), (y,z) and (x,y) planes, needs matplotlib.
        points with positive contrast are drawn in color, zero contrast in grey and negative contrast in black

        path   : file to save the figure to, which is then closed; None returns the figure instead
        color  : color of the points with positive contrast
        extent : the axes go from -1.1 * extent to 1.1 * extent [A]; by default the extent of these points.
                 give the largest extent of several point distributions to plot them on the same scale
        """
        from .plots import plot_projections
        return plot_projections(self.x, self.y, self.z, self.sld, path, color=color, extent=self.extent if extent is None else extent)


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


class Particle:
    """
    a particle built from geometric subunits

    give its subunits to create_particle, or add them with add_subunit, then fill it with random points with generate_points,
    which gives the pair distance distribution and the scattering.
    """

    def __init__(self, exclude_overlap):
        self.exclude_overlap = exclude_overlap
        self._placements: list[Placement] = []

    def add_subunit(self, subunit: Subunit | Placement, /) -> None:
        """
        add a subunit to the particle: a subunit, e.g. subunits.Cylinder(radius=20, length=100), at the origin with unit contrast,
        or a placed subunit, e.g. subunits.Cylinder(radius=20, length=100).placed(com=(0, 0, 50), sld=2)

        point distributions generated before the subunit is added are not affected. raises ValueError on anything else
        """
        if not isinstance(subunit, (Subunit, Placement)):
            raise InputError("add_subunit takes a subunit, e.g. Sphere(radius=50), or a placed subunit, e.g. "
                             "Sphere(radius=50).placed(com=(0, 0, 60)), but %r was given" % (subunit,))
        self._placements.append(subunit if isinstance(subunit, Placement) else subunit.placed())

    def generate_points(self, Npoints: int, *, rng: np.random.Generator | int | None = None, prpoints: int = 100,
                        polydispersity: float = 0.0, use_ausaxs: bool = True) -> PointDistribution:
        """
        fill the particle with random points

        Npoints        : roughly how many points fill the particle, split between the subunits by their volume
        rng            : random number generator (np.random.Generator), or a seed for one; None draws a fresh seed
        prpoints       : number of bins in p(r)
        polydispersity : relative polydispersity of the particle size
        use_ausaxs     : try the AUSAXS backend first for the histogram, and fall back to the default method if pyausaxs is not
                         installed or fails

        every call draws new points, unless the same seed is given. raises ValueError if the particle has no subunits
        """
        if not self._placements:
            raise InputError("the particle has no subunits; add them with add_subunit")
        distribution = GenerateAllPoints(Npoints, list(self._placements), self.exclude_overlap,
                                         np.random.default_rng(rng)).onGeneratingAllPointsSeparately()
        return PointDistribution(distribution, prpoints, polydispersity, use_ausaxs)


def create_particle(*subunits: Subunit | Placement, exclude_overlap: bool = True) -> Particle:
    """
    create a particle from any number of subunits; more can be added with add_subunit. fill it with points with generate_points

    subunits        : subunits, e.g. subunits.Cylinder(radius=20, length=100), at the origin with unit contrast, or
                      placed subunits, e.g. subunits.Cylinder(radius=20, length=100).placed(com=(0, 0, 50), sld=2)
    exclude_overlap : remove the points of a subunit that lie inside earlier subunits

    raises ValueError on anything else
    """
    particle = Particle(exclude_overlap)
    for subunit in subunits:
        particle.add_subunit(subunit)
    return particle
