"""Filling the subunits of a particle with points: placement, rotation and overlap exclusion."""

import logging

import numpy as np

from ..rotation import rotate_and_translate, undo_rotate_and_translate
from ..subunits.Subunit import Placement
from .ModelPointDistribution import ModelPointDistribution

logger = logging.getLogger(__name__)

class GenerateAllPoints:
    def __init__(self, Npoints, placements: list[Placement], exclude_overlap, rng: np.random.Generator):
        self.Npoints = Npoints
        self.rng = rng
        self.subunits = [p.subunit for p in placements]
        self.Number_of_subunits = len(placements)
        self.sld = [p.sld for p in placements]
        self.com = [p.com for p in placements]
        self.rotation = [p.rotation for p in placements]
        self.rotation_points = [p.rotation_point for p in placements]
        self.exclude_overlap = exclude_overlap

    @staticmethod
    def AppendingPoints(x_new, y_new, z_new,sld_new, x_add, y_add, z_add, sld_add):
        """append new points to vectors of point coordinates"""
        
        # add points to (x_new,y_new,z_new)
        if isinstance(x_new, int):
            # if these are the first points to append to (x_new,y_new,z_new)
            x_new = x_add
            y_new = y_add
            z_new = z_add
            sld_new = sld_add
        else:
            x_new = np.append(x_new, x_add)
            y_new = np.append(y_new, y_add)
            z_new = np.append(z_new, z_add)
            sld_new = np.append(sld_new, sld_add)

        return x_new, y_new, z_new, sld_new

    @staticmethod
    def onCheckOverlap(x, y, z, p, rotation, rotation_point, com, subunit):
        """
        check for overlap with previous subunits. 
        if overlap, the point is removed
        """
        # undo the rotation and translation of the subunit being checked against,
        # so the points are expressed in that subunit's own frame
        x_eff, y_eff, z_eff = undo_rotate_and_translate(x, y, z, rotation, rotation_point, com)

        # then keep the points outside it
        keep = subunit.outside(x_eff, y_eff, z_eff)
        x_add, y_add, z_add, sld_add = x[keep], y[keep], z[keep], p[keep]

        ## number of excluded points
        N_x = len(x) - int(np.count_nonzero(keep))
        return x_add, y_add, z_add, sld_add, N_x

    def onGeneratingAllPointsSeparately(self):
        """Generating points for all subunits from each built model, but
        save them separately in their own list"""
        volume = []
        sum_vol = 0

        #Get volume of each subunit
        for i in range(self.Number_of_subunits):
            v = self.subunits[i].volume()
            volume.append(v)
            sum_vol += v

        N, rho, N_exclude = [], [], []
        x_new, y_new, z_new, sld_new, volume_total = [], [], [], [], 0

        for i in range(self.Number_of_subunits):
            Npoints = int(self.Npoints * volume[i] / sum_vol)
            
            x_add, y_add, z_add = self.subunits[i].generate_points(Npoints, self.rng)

            # rotate and translate
            x_add, y_add, z_add = rotate_and_translate(x_add, y_add, z_add, self.rotation[i], self.rotation_points[i], self.com[i])
            
            #Remaining points
            N_subunit = len(x_add)
            rho_subunit = N_subunit / volume[i]
            sld_add = np.ones(N_subunit) * self.sld[i]

            #Check for overlap with previous subunits
            N_x_sum = 0
            if self.exclude_overlap:
                for j in range(i): 
                    x_add, y_add, z_add, sld_add, N_x = self.onCheckOverlap(x_add, y_add, z_add, sld_add, self.rotation[j],
                                                    self.rotation_points[j], self.com[j], self.subunits[j])
                    N_x_sum += N_x
    
            N.append(N_subunit)
            rho.append(rho_subunit)
            N_exclude.append(N_x_sum)
            fraction_left = (N_subunit-N_x_sum) / max(N_subunit, 1)
            volume_total += volume[i] * fraction_left

            x_new.append(x_add)
            y_new.append(y_add)
            z_new.append(z_add)
            sld_new.append(sld_add)
        
        #Show information about the model and its subunits
        N_remain = []
        for j in range(self.Number_of_subunits):
            srho = rho[j] * self.sld[j]
            N_remain.append(N[j] - N_exclude[j])
            logger.info(f"        {N[j]} points for subunit {j}: {self.subunits[j].aliases[0]}")
            logger.info(f"             Point density     : {rho[j]:.3e} (points per volume)")
            logger.info(f"             Scattering density: {srho:.3e} (density times scattering length)")
            if self.exclude_overlap:
                logger.info(f"             Excluded points   : {N_exclude[j]} (overlap region)")
            else:
                logger.info(f"             Excluded points   : none - exclude overlap disabled")
            logger.info(f"             Remaining points  : {N_remain[j]} (non-overlapping region)")
        N_total = sum(N_remain)
        logger.info(f"        Total points in model: {N_total}")
        logger.info(f"        Total volume of model: {volume_total:.3e} A^3")
        logger.info(" ")

        return ModelPointDistribution(x=x_new, y=y_new, z=z_new, sld=sld_new, volume_total=volume_total)
