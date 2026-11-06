import numpy as np

from ..helpfunctions import InputError

def check_dimension(name,dimensions,n):
    """check if the number of input dimensions for the subunit is correct, else raise an InputError"""
    len_dim = len(dimensions)
    if len_dim != n:
            dim = ' dimension ' if n == 1 else ' dimensions '
            were = ' was ' if len_dim == 1 else ' were '
            raise InputError("subunit " + name + " needs " + str(n) + dim + "(provided after --dimensions or -d), but " + str(len_dim) + were + "given: " + str(dimensions))

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

class Subunit:
    """
    A body placed in the particle, painting its SLD onto the points inside it.

    A subunit describes its body in its own frame, centred at the origin, by
    implementing:
        getVolume()     : volume of the body
        getHalfWidths() : half-widths of a box around the body
        checkOverlap()  : indices of the points OUTSIDE the body

    and passes on to this class where the body is, and what it is made of:
        com            : displacement [A], default (0, 0, 0)
        rotation       : Euler angles (intrinsic ZYX) [degrees], default (0, 0, 0)
        rotation_point : point to rotate around [A], default (0, 0, 0)
        sld            : excess scattering length density (contrast) [1e-6 A^-2], default 1
    """

    def __init__(self, com=(0, 0, 0), rotation=(0, 0, 0), rotation_point=(0, 0, 0), sld=1.0):
        self.com = com
        self.rotation = rotation
        self.rotation_point = rotation_point
        self.sld = sld

    def getBoundingBox(self):
        """axis-aligned box [lower, upper] around the placed body"""
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1)
        corners = corners * np.asarray(self.getHalfWidths(), dtype=float)[:, np.newaxis]
        corners = np.vstack(rotate_and_translate(*corners, self.rotation, self.rotation_point, self.com))
        return corners.min(axis=1), corners.max(axis=1)

    def contains(self, x, y, z):
        """mask of the points that lie inside the placed body"""
        x_eff, y_eff, z_eff = undo_rotate_and_translate(x, y, z, self.rotation, self.rotation_point, self.com)
        inside = np.ones(len(x), dtype=bool)
        inside[self.checkOverlap(x_eff, y_eff, z_eff)] = False
        return inside
