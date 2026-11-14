"""Rotation and translation of points by Euler angles."""

import numpy as np


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
    subunit, so that outside() can be applied to them.

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
