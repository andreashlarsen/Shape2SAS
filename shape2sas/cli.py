#!/usr/bin/python3

import time
import argparse
import logging
import sys
import numpy as np
import shutil

from . import __version__
from .helpfunctions import (
    check_3Dinput,
    check_input,
    float_list,
    get_header_footer,
    getStructureFactorClass,
    InputError,
    separate_string,
    str2bool,
)
from .models import save_points
from .particle import create_particle
from .theoretical_scattering import save_I_func, save_pr_func, uniform_q
from .structure_factors.structure_factors_helpfunctions import save_S_func
from .simulated_scattering import save_Isim_func, simulate_scattering
from .plots import generate_pdb, plot_2D, plot_fit, plot_results
from .sesans import calc_G_sesans, simulate_sesans
from .sesans.sesans_output import plot_sesans, save_sesans

# named explicitly, so the messages also reach the handlers when run as python -m shape2sas.cli
logger = logging.getLogger("shape2sas.cli")


def log_to_terminal_and_file():
    """send the progress messages of the CLI and the library to the console and to a new shape2sas.log"""
    package_logger = logging.getLogger("shape2sas")
    package_logger.setLevel(logging.INFO)
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
        handler.close()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler('shape2sas.log', mode='w')):
        handler.setFormatter(logging.Formatter('%(message)s'))
        package_logger.addHandler(handler)


def main(argv=None):
    """Entry point for the ``shape2sas`` command."""
    log_to_terminal_and_file()
    try:
        run(argv)
    except InputError as e:
        logger.error("\nERROR: %s\n" % e)
        sys.exit(1)


def run(argv=None):
    """run the simulations requested on the command line"""
    argv = sys.argv[1:] if argv is None else list(argv)
       
    ### start timing
    start_total = time.time()

    ### welcome message
    logger.info('#######################################################################################')
    logger.info('RUNNING shape2sas version %s \n - for instructions type: shape2sas -h' % __version__)
    command = "shape2sas"
    for aa in argv:
        if ' ' in aa:
            command += " \"%s\"" % aa
        else:
            command += " %s" % aa
    logger.info('command used: %s' % command)
    logger.info('#######################################################################################')

    ### input values 
    parser = argparse.ArgumentParser(description='Shape2SaS - calculates small-angle scattering from a given shape defined by the user.')
      
    # mandatory inputs
    parser.add_argument('-s', '--subunit', type=separate_string, nargs='+', action='extend',
                       help='Type of subunits for each model.')
    parser.add_argument('-d', '--dimension', type=float_list, nargs='+', action='append',
                        help='dimensions of subunits for each model.')
    
    # optional model-dependent inputs:
    parser.add_argument('-m', '--model_name', nargs='+', action='extend',
                        help='Name of model.')
    parser.add_argument('-sld', '--sld', type=float, nargs='+', action='append',
                        help='excess scattering length density or contrast.')
    parser.add_argument('-pd', '--polydispersity', type=float, nargs='+', action='extend',
                        help='Polydispersity of subunits for each model.')
    parser.add_argument('-com', '--com', type=float_list, nargs='+', action='append',
                         help='displacement for each subunits in each model.')
    parser.add_argument('-rot', '--rotation', type=float_list, nargs='+', action='append',
                         help='rotation for each subunits in each model.')
    parser.add_argument('-rotp', '--rotation_points', type=float_list, nargs='+', action='append',
                         help='point to rotate around, for each subunit in each model (default: the subunit centre).')
    parser.add_argument('-sigmar', '--sigma_r', type=float, nargs='+', action='extend',
                        help='interface roughness for each model.')
    parser.add_argument('-c', '--conc', type=float, nargs='+', action='extend',
                        help='volume fraction concentration.')
    parser.add_argument('-exclude', '--exclude_overlap', type=str2bool, nargs='+', action='extend',
                        help='bool to exclude overlap.')
    parser.add_argument('-noausaxs', '--disable_ausaxs', action='store_true', default=False,
                        help='include flag (no input) to disable the AUSAXS backend (pyausaxs) for the pair distance calculation, and always use the default method.')

    # optional structure factor related inputs
    parser.add_argument('-S', '--S', type=str, nargs='+', action='extend',
                        help='structure factor: None/HS/aggregation in each model.')
    parser.add_argument('-Sp', '--S_par', type=float_list, nargs='+', action='append',
                        help='parameters of structure factor for each model.')

    # optional general inputs
    parser.add_argument('-qmin', '--qmin', type=float, default=0.001, 
                        help='Minimum q-value for the scattering curve.')
    parser.add_argument('-qmax', '--qmax', type=float, default=0.5, 
                        help='Maximum q-value for the scattering curve.')
    parser.add_argument('-Nq', '--qpoints', type=int, default=400, 
                        help='Number of points in q.')
    parser.add_argument('-Np', '--prpoints', type=int, default=100, 
                        help='Number of points in the pair distance distribution function.')
    parser.add_argument('-N', '--Npoints', type=int, default=8000, 
                        help='Number of simulated points per model.')
    parser.add_argument('-expo', '--exposure', type=float, default=500, 
                        help='Exposure time in arbitrary units.')

    # optional plot options
    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, 
                        help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-hres', '--high_res', action='store_true', default=False, 
                        help='include flag (no input) to output high resolution plot.')

    # optional SESANS-related options (Shape2SESANS)
    parser.add_argument('-ss', '--sesans', action='store_true', default=False,
                        help='Calculate SESANS data from the SAS data.')
    parser.add_argument('-sse', '--sesans_error', type=float, default=0.02, 
                        help='Baseline SESANS error relative to max signal.')
    parser.add_argument('-Nd', '--deltapoints', type=int, default=150,
                        help='Number of points in delta.')
    
    # optional experimental data-related options (Shape2SAS-fit)
    parser.add_argument('-dat','--data', 
                        help='Path to experimental data')
    
    args = parser.parse_args(argv)

    ### check input 
  
    # check that subunits and dimensions are provided
    if args.subunit is None:
        raise argparse.ArgumentError(args.subunit, "No subunit type was given as an input.")  
    if args.dimension is None:
        raise argparse.ArgumentError(args.dimension, "No dimensions were given as an input.")
    # check that number of subunits matches number of dimension lists
    for subunit, dimension in zip(args.subunit, args.dimension):
         if len(subunit) != len(dimension):
            raise argparse.ArgumentTypeError("Mismatch between number subunit types (%d) and dimensions lists (%d)." % (len(subunit),len(dimension)))    
    num_models = len(args.subunit)
    if num_models == 1:
        logger.info(f"Simulating {num_models} model...")
    else: 
        logger.info(f"Simulating {num_models} models...")

    # prepare lists (several models can be simulated simultaneously)
    r_list, pr_norm_list, I_list, I_sim_list, sigma_list, S_list = [], [], [], [], [], [] 
    x_list, y_list, z_list, sld_list, model_filename_list, model_name_list = [], [], [], [], [], []
    if args.sesans:
        delta_list,G_list,G_sim_list,sigma_G_list = [],[],[],[]

    # loop over models
    for i in range(num_models):
        
        ### read and print model name for model i
        model_name = check_input(args.model_name, f"Model {i}", "model name", i)
        if model_name in model_name_list:
            #model names should be unique - else add a number
            model_name += '_' + str(i+1)  
        model_filename = "_".join(model_name.split()) # remove whitespace for filenames
        model_name_list.append(model_name)
        model_filename_list.append(model_filename)

        #### read number of subunits, SLD, COM, rotation and exclude overlap for model i
        N_subunits = len(args.subunit[i])
        sld = check_3Dinput(args.sld, [1.0], "SLD", N_subunits, i)
        com = check_3Dinput(args.com, [[0, 0, 0]], "COM", N_subunits, i)
        rotation = check_3Dinput(args.rotation, [[0, 0, 0]], "rotation", N_subunits, i)
        rotation_points = check_3Dinput(args.rotation_points, [[0, 0, 0]], "rotation points", N_subunits, i)
        exclude_overlap = check_input(args.exclude_overlap, True, "exclude_overlap", i)
        polydispersity = check_input(args.polydispersity, 0.0, "polydispersity", i)

        ### make point cloud
        logger.info(f"    Generating points for Model: " + model_name)
        particle = create_particle(exclude_overlap=exclude_overlap)
        for j in range(N_subunits):
            particle.add_subunit(args.subunit[i][j], args.dimension[i][j], sld=sld[j], com=com[j],
                                 rotation=rotation[j], rotation_point=rotation_points[j])
        points = particle.generate_points(args.Npoints, prpoints=args.prpoints, polydispersity=polydispersity,
                                          use_ausaxs=not args.disable_ausaxs)
        x, y, z, sld_points = points.x, points.y, points.z, points.sld
        save_points(x, y, z, sld_points, model_filename)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        sld_list.append(sld_points)

        ### read concentration, interface roughness/fuzziness, structure factor and structure factor-related parameters for model i
        conc = check_input(args.conc, 0.02, "concentration", i)
        sigma_r = check_input(args.sigma_r, 0.0, "sigma_r", i)
        S_type = check_input(args.S, 'None', "Structure type", i)
        try:
            S_par = args.S_par[i][0]
        except:
            S_par = []

        ### calculate p(r)
        logger.info("\n    Calculating pair distance distribution, p(r)...")
        r, pr = points.histogram.r, points.histogram.pr
        save_pr_func(r, pr, model_filename)
        r_list.append(r)
        pr_norm_list.append(pr)

        ### define q (and if sesans is opted for, also define the spin echo length, delta)
        if args.sesans:
            # make extended q-range for sesans
            # the structure factor decides the length scale: an aggregate is
            # larger than the particle it is built from
            qmin, deltamax = getStructureFactorClass(S_type).getSesansRange(S_par, points.histogram.dmax)
            qmax = 1e4 * qmin
            qpoints = 5000
            q = np.linspace(qmin,qmax,qpoints)
            delta = np.linspace(0, deltamax, args.deltapoints)
        elif args.data:
            header,footer = get_header_footer(args.data) # replace with function
            q,I_exp,sigma_exp = np.genfromtxt(args.data,skip_header=header,skip_footer=footer,usecols=[0,1,2],unpack=True)
        else:
            q = uniform_q(args.qmin,args.qmax,args.qpoints)

        logger.info("\n    Calculating intensity, I(q)...")

        ### calculate theoretical SAS, in absolute units, with its structure factor (and SESANS)
        S_eff = points.structure_factor(q, S_type, S_par)
        save_S_func(q,S_eff,model_filename)
        S_list.append(S_eff)
        I = points.scattering(q, conc, S_type, S_par, sigma_r)
        save_I_func(q,I,model_filename)
        I_list.append(I)
        if args.sesans:
            # calculated theoretical SESANS
            G = calc_G_sesans(q,delta,I)
            delta_list.append(delta)
            G_list.append(G)

        ### simulate SAXS (and SESANS)
        I_sim, sigma = simulate_scattering(q,I,args.exposure)
        save_Isim_func(q,I_sim,sigma,model_filename)
        I_sim_list.append(I_sim)
        sigma_list.append(sigma)
        if args.sesans:
            # simulate sesans data         
            G_sim,sigma_G = simulate_sesans(delta,G,args.sesans_error)
            # append to list (in case of multiple models)
            G_sim_list.append(G_sim)
            sigma_G_list.append(sigma_G)   

    logger.info(" ")
    logger.info("Generating plots")
    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forestgreen','gold','darkred','coral','peru','olive','springgreen','teal','skyblue','navy','lavender','blueviolet','deeppink']

    if args.high_res:
        filetype = 'pdf'
    else:
        filetype = 'png'

    # plot 2D projections
    for m in model_filename_list:
        print("    2D projection:       points_" + m + "." + filetype)
    plot_2D(x_list, y_list, z_list, sld_list, model_filename_list, filetype, colors)
    
    # 3D vizualization: generate pdb file with points
    for m in model_filename_list:
        print("    3D models:           " + m + ".pdb")
    generate_pdb(x_list, y_list, z_list, sld_list, model_filename_list)
    
    # plot p(r) and I(q)
    print("    pr, Iq, and Isim:    plot." + filetype)
    plot_results(q, r_list, pr_norm_list, I_list, I_sim_list, sigma_list, S_list, model_name_list, args.xscale_lin, filetype, colors)

    # plot fit
    if args.data:
        data_filename = args.data.split('/')[-1]
        print("    fit(s) to exp data : fit." + filetype)
        plot_fit(q, I_list, I_exp, sigma_exp, model_name_list, data_filename, args.xscale_lin, filetype, colors)

    # plot and save sesans 
    if args.sesans:
        print("    SESANS G and Gsim  : sesans." + filetype)
        plot_sesans(delta_list, G_list, G_sim_list, sigma_G_list, model_name_list, filetype, colors)
        save_sesans(delta_list, G_list, G_sim_list, sigma_G_list, model_filename_list)

    time_total = time.time() - start_total
    logger.info(" ")
    logger.info("Simulation successfully completed.")
    logger.info("    Total run time:      " + str(round(time_total, 1)) + " seconds.")
    logger.info(" ")

    # close log file and copy into model directories
    #f_out.close()
    extension = '.' + filetype
    for model_filename in model_filename_list:
        shutil.copy('shape2sas.log', model_filename + '/' + model_filename + '.log' )
        shutil.copy('plot' + extension, model_filename + '/plot_' + model_filename + extension)
        if args.data:
            shutil.copy('fit' + extension, model_filename + '/fit_' + model_filename + '_' + data_filename + extension)
        if args.sesans:
            shutil.copy('sesans' + extension, model_filename + '/sesans_' + model_filename + extension )


if __name__ == "__main__":
    main()
