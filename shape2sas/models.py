"""Building a point model: subunit placement, rotation and overlap exclusion."""

from dataclasses import dataclass

import logging
import os

import numpy as np

from .subunits.Subunit import Subunit

logger = logging.getLogger(__name__)

@dataclass
class ModelPointDistribution:
    """
    Point distribution of a model
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sld: np.ndarray #scattering length density for each point
    volume_total: float

@dataclass
class SubunitPlacement:
    """a subunit of a particle, and where it is placed"""

    subunit: Subunit
    sld: float
    com: list[float]
    rotation: list[float]
    rotation_point: list[float]

def euler_rotation_matrix(alpha,beta,gamma):
    """
    Rotation matrix from Euler angles, intrinsic ZYX convention.
    Input angles in radians.
    """
    ca,cb,cg = np.cos(alpha),np.cos(beta),np.cos(gamma)
    sa,sb,sg = np.sin(alpha),np.sin(beta),np.sin(gamma)
    return np.array([
        [cb*cg, sa*sb*cg - ca*sg, ca*sb*cg + sa*sg],
        [cb*sg, sa*sb*sg + ca*cg, ca*sb*sg - sa*cg],
        [-sb  , sa*cb           , ca*cb           ]
    ])

def rotate_and_translate(x,y,z,rotation,rotation_point,com):
    """
    Rotate points around rotation_point, then translate them by com:

        v' = R * (v - rp) + rp + com

    input angles in degrees
    """
    R = euler_rotation_matrix(*np.radians(rotation))
    rp = np.asarray(rotation_point,dtype=float)
    T = np.asarray(com,dtype=float)
    offset = rp + T - np.dot(R,rp)
    out = np.dot(R,np.vstack([x,y,z])) + offset[:,np.newaxis]
    return out[0],out[1],out[2]

def undo_rotate_and_translate(x,y,z,rotation,rotation_point,com):
    """
    Inverse of rotate_and_translate(): bring points back into the frame of the
    subunit, so that checkOverlap() can be applied to them.

        v = R^T * (v' - com - rp) + rp

    NOTE: the inverse rotation is the TRANSPOSE of R. It is *not* the same as
    rotating by (-alpha,-beta,-gamma), which is only equivalent when at most
    one of the three angles is non-zero.

    input angles in degrees
    """
    R_inv = euler_rotation_matrix(*np.radians(rotation)).T
    rp = np.asarray(rotation_point,dtype=float)
    T = np.asarray(com,dtype=float)
    out = np.dot(R_inv,np.vstack([x,y,z]) - (T+rp)[:,np.newaxis]) + rp[:,np.newaxis]
    return out[0],out[1],out[2]

class GenerateAllPoints:
    def __init__(self, Npoints, placements, exclude_overlap):
        self.Npoints = Npoints
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

        # then check overlaps
        idx = subunit.checkOverlap(x_eff, y_eff, z_eff)
        x_add, y_add, z_add, sld_add = x[idx], y[idx], z[idx], p[idx]

        ## number of excluded points
        N_x = len(x) - len(idx[0])
        return x_add, y_add, z_add, sld_add, N_x

    def onGeneratingAllPointsSeparately(self):
        """Generating points for all subunits from each built model, but
        save them separately in their own list"""
        volume = []
        sum_vol = 0

        #Get volume of each subunit
        for i in range(self.Number_of_subunits):
            v = self.subunits[i].getVolume()
            volume.append(v)
            sum_vol += v

        N, rho, N_exclude = [], [], []
        x_new, y_new, z_new, sld_new, volume_total = [], [], [], [], 0

        for i in range(self.Number_of_subunits):
            Npoints = int(self.Npoints * volume[i] / sum_vol)
            
            x_add, y_add, z_add = self.subunits[i].getPointDistribution(Npoints)

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

def save_points(x, y, z, sld, model_filename):
    """save point cloud to a file"""
    os.makedirs(model_filename, exist_ok=True)
    with open('%s/points_%s.txt' % (model_filename,model_filename),'w') as f:
        f.write('# x y z sld\n')
        for xi,yi,zi,s in zip(x,y,z,sld):
            f.write('%f %f %f %f\n' % (xi,yi,zi,s))
