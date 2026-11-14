"""Theoretical scattering: the intensity I(q) from the form factor and structure factor, q values, and saving to text files."""

import os

import numpy as np

def calc_Iq_func(q, Pq, S_eff, sigma_r):
    """
    calculates intensity
    """

    ## multiply form factor with structure factor
    I = Pq * S_eff

    ## interface roughness (Skar-Gislinge et al. 2011, DOI: 10.1039/c0cp01074j)
    if sigma_r > 0.0:
        roughness = np.exp(-(q * sigma_r)**2 / 2)
        I *= roughness

    return I
    
def uniform_q(qmin=0.001, qmax=0.5, qpoints=400):
    """equidistant q values from qmin to qmax [1/A]"""
    return np.linspace(qmin, qmax, qpoints)

def write_columns(path, names, columns, title=None):
    """save columns of numbers as a text file, with the column names (and a title, of one or more lines) in the header"""
    with open(path, 'w') as f:
        if title is not None:
            for line in title.splitlines():
                f.write('# %s\n' % line)
        f.write('# ' + ' '.join('%-12s' % name for name in names) + '\n')
        for row in zip(*columns):
            f.write('  ' + ' '.join('%-12.5e' % value for value in row) + '\n')

def save_pr_func(r,pr,model_filename):
    """save pair distance distribution p(r) to <model_filename>/pr_<model_filename>.dat"""
    os.makedirs(model_filename, exist_ok=True)
    write_columns('%s/pr_%s.dat' % (model_filename,model_filename), ('r','p(r)'), (r, pr))

def save_I_func(q, I, model_filename):
    """save theoretical intensity to <model_filename>/Iq_<model_filename>.dat"""
    os.makedirs(model_filename, exist_ok=True)
    write_columns('%s/Iq_%s.dat' % (model_filename,model_filename), ('q','I'), (q, I), title='Theoretical SAS data')
