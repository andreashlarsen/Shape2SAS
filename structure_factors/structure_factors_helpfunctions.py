import numpy as np

def printt(s): 
    """ print and write to log file"""
    print(s)
    with open('shape2sas.log','a') as f:
        f.write('%s\n' %s)

def sinc(x):
    """
    function for calculating sinc = sin(x)/x
    numpy.sinc is defined as sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.sinc(x / np.pi)

def check_Spar(name,S_par,par_names):
    """check if the number of input parameters for the structure factor is correct, else return error message"""
    n = len(par_names)
    len_par = len(S_par)
    if len_par != n:
        par = ' parameter ' if n == 1 else ' parameters '
        were = ' was ' if len_par == 1 else ' were '
        printt("\nERROR: structure factor " + name + " needs " + str(n) + par + "(provided after --S_par or -Sp): " + ", ".join(par_names) + ", but " + str(len_par) + were + "given: " + str(list(S_par)) + "\n")
        exit()

def calc_com_dist(point_distribution):
    """ 
    calc contrast-weighted com distance

    the coordinates are stored per subunit, so they are concatenated first:
    the subunits generally contain different numbers of points, and numpy
    cannot average over such a ragged list of arrays
    """
    x = np.concatenate(point_distribution.x)
    y = np.concatenate(point_distribution.y)
    z = np.concatenate(point_distribution.z)
    w = np.abs(np.concatenate(point_distribution.sld))

    if np.sum(w) == 0:
        w = np.ones(len(x))

    x_com, y_com, z_com = np.average(x, weights=w), np.average(y, weights=w), np.average(z, weights=w)
    dx, dy, dz = x - x_com, y - y_com, z - z_com
    com_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    return com_dist

def calc_A00(q,point_distribution):
    """
    calc zeroth order sph harm, for decoupling approximation
    """
    d = calc_com_dist(point_distribution).astype(np.float32, copy=False)
    M = len(q)
    A00 = np.zeros(M)
    sld = np.concatenate(point_distribution.sld).astype(np.float32, copy=False)
    for i in range(M):
        qr = q[i] * d
        A00[i] = np.sum(sld * sinc(qr))
    A00 = A00 / A00[0] # normalise, A00[0] = 1

    return A00

def decoupling_approx(q,point_distribution,Pq,S):
    """
    modify structure factor with the decoupling approximation
    for combining structure factors with non-spherical (or polydisperse) particles

    see, for example, Larsen et al 2020: https://doi.org/10.1107/S1600576720006500
    and refs therein

    input
    q
    point_distribution : coordinates and contrasts
    Pq                 : form factor
    S                  : structure factor

    output
    S_eff              : effective structure factor, after applying decoupl. approx
    """
    A00 = calc_A00(q,point_distribution)
    const = 1e-3 # add constant in nominator and denominator, for stability (numerical errors for small values dampened)
    Beta = (A00**2 + const) / (Pq + const)
    S_eff = 1 + Beta * (S - 1)
    return S_eff

def default_sesans_range(dmax):
    """default spin echo length range, set by the size of the particle itself"""
    qmin = 0.001 * np.pi / dmax
    deltamax = 3 * dmax
    return qmin, deltamax
