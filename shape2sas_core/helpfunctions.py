"""General helpers shared across Shape2SAS.

Only non-specific helpers belong here. The calculations themselves live in
their own modules: models.py, theoretical_scattering.py,
simulated_scattering.py, plots.py, and the subunits, structure_factors and
sesans subpackages.
"""

import argparse
import inspect
import re
import warnings

import numpy as np

def normalise_alias(name):
    """normalise a name so that 'Hollow sphere', 'hollow_sphere' and
    'hollowsphere' all refer to the same thing"""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")

def build_alias_registry(package):
    """map every alias of every class in a package to that class"""
    registry = {}
    for _, cls in inspect.getmembers(package, inspect.isclass):
        for alias in getattr(cls, "aliases", []):
            registry[normalise_alias(alias)] = cls
    return registry

def lookup_alias(registry, name, what):
    """look up a class by any of its aliases, or list the valid names"""
    try:
        return registry[normalise_alias(name)]
    except KeyError:
        available = sorted(set(cls.aliases[0] for cls in registry.values()))
        printt("\nERROR: unknown " + what + " '" + str(name) + "'. Available " + what + "s: " + ", ".join(available) + "\n")
        exit()

def getStructureFactorClass(stype):
    """look up a structure factor class by any of its aliases"""
    from shape2sas_core import structure_factors
    return lookup_alias(build_alias_registry(structure_factors), stype, "structure factor")

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

def get_header_footer(file):
    """Count header and footer lines with non-numeric entries."""
    header, footer = 0, 0
    with open(file, errors='ignore') as f:
        lines = f.readlines()

    CONTINUE_H, CONTINUE_F = True, True
    j = 0
    while CONTINUE_H or CONTINUE_F:
        line_h, line_f = lines[j], lines[-1-j]
        tmp_h, tmp_f = line_h.split(), line_f.split()
        if CONTINUE_H:
            try:
                for val in tmp_h[:3]:
                    float(val)
                CONTINUE_H = False
            except:
                header += 1
        if CONTINUE_F:
            try:
                for val in tmp_f[:3]:
                    float(val)
                CONTINUE_F = False
            except:
                footer += 1
        j += 1
    return header, footer

def str2bool(v):
    """
    Function to circumvent the argparse default behaviour 
    of not taking False inputs, when default=True.
    """
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def separate_string(arg):
    arg = re.split('[ ,]+', arg)
    return [str(i) for i in arg]

def float_list(arg):
    """
    Function to convert a string to a list of floats.
    Note that this function can interpret numbers with scientific notation 
    and negative numbers.

    input:
        arg: string, input string

    output:
        list of floats
    """
    arg = re.sub(r'\s+', ' ', arg.strip())
    arg = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", arg)
    return [float(i) for i in arg]

def check_3Dinput(input: list, default: list, name: str, N_subunits: int, i: int):
    """
    Function to check if 3D vector input matches 
    in lenght with the number of subunits

    input:
        input: list of floats, input values
        default: list of floats, default values

    output:
        list of floats
    """
    try:
        inputted = input[i]
        if len(inputted) != N_subunits:
            warnings.warn(f"The number of subunits and {name} do not match. Using {default}")
            inputted = default * N_subunits
    except:
        inputted = default * N_subunits
        #warnings.warn(f"Could not find {name}. Using default {default}.")

    return inputted

def check_input(input: float, default: float, name: str, i: int):
    """
    Function to check if input is given, 
    if not, use default value.

    input:
        input: float, input value
        default: float, default value
        name: string, name of the input

    output:
        float
    """
    try:
        inputted = input[i]
    except:
        inputted = default
        #warnings.warn(f"Could not find {name}. Using default {default}.")

    return inputted

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