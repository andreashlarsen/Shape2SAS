
import time
import argparse
from sys import argv
import numpy as np
from helpfunctions import *

# current version
version = 2.2

if __name__ == "__main__":
       
    ### make log file and define printt function
    f_out = open('shape2sas.log','w')
    def printt(s): 
        """ print and write to log file"""
        print(s)
        f_out.write('%s\n' %s)
    
    ### welcome message
    printt('#######################################################################################')
    printt('RUNNING shape2sas.py version %s \n - for instructions type: python shape2sas.py -h' % version)
    command = "python"
    for aa in argv:
        if ' ' in aa:
            command += " \"%s\"" % aa
        else:
            command += " %s" %aa
    printt('command used: %s' % command)
    printt('#######################################################################################')

    start_total = time.time()

    ### input values 
    parser = argparse.ArgumentParser(description='Shape2SaS - calculates small-angle scattering from a given shape defined by the user.')
      
    # Mandatory inputs
    parser.add_argument('-subunit', '--subunit_type', type=separate_string, nargs='+', action='extend',
                       help='Type of subunits for each model.')
    parser.add_argument('-dim', '--dimension', type=float_list, nargs='+', action='append',
                        help='dimensions of subunits for each model.')
    
    # Optional model-dependent inputs:
    parser.add_argument('-modelname', '--model_name', nargs='+', action='extend',
                        help='Name of model.')
    parser.add_argument('-sld', '--sld', type=float, nargs='+', action='append',
                        help='excess scattering length density or contrast.')
    parser.add_argument('-poly', '--polydispersity', type=float, nargs='+', action='extend',
                        help='Polydispersity of subunits for each model.')
    parser.add_argument('-com', '--com', type=float_list, nargs='+', action='append', 
                        help='displacement for each subunits in each model.')
    parser.add_argument('-rotation', '--rotation', type=float_list, nargs='+', action='append', 
                        help='rotation for each subunits in each model.')
    parser.add_argument('-sigmar', '--sigma_r', type=float, nargs='+', action='extend',
                        help='interface roughness for each model.')
    parser.add_argument('-conc', '--conc', type=float, nargs='+', action='extend',
                        help='volume fraction concentration.')
    parser.add_argument('-exclude', '--exclude_overlap', type=str2bool, nargs='+', action='extend', 
                        help='bool to exclude overlap.')

    # Optional structure factor related inputs
    parser.add_argument('-S', '--S', type=str, nargs='+', action='extend',
                        help='structure factor: None/HS/aggregation in each model.')
    parser.add_argument('-rhs', '--r_hs', type=float, nargs='+', action='extend',
                        help='radius of hard sphere in each model.')
    parser.add_argument('-frac', '--frac', type=float, nargs='+', action='extend',
                        help='fraction of particles in aggregated form for each model.')
    parser.add_argument('-Naggr', '--N_aggr', type=int, nargs='+', action='extend',
                        help='Number of particles per aggregate for each model.')
    parser.add_argument('-Reff', '--R_eff', type=float, nargs='+', action='extend',
                        help='Effective radius of aggregates for each model.')
    #parser.add_argument('-par', '--par', type=float, nargs='+', action='extend',
    #                    help='structure factor parameters.')

    # Optional general inputs
    parser.add_argument('-qmin', '--qmin', type=float, default=SimulationParameters.qmin, 
                        help='Minimum q-value for the scattering curve.')
    parser.add_argument('-qmax', '--qmax', type=float, default=SimulationParameters.qmax, 
                        help='Maximum q-value for the scattering curve.')
    parser.add_argument('-Nq', '--qpoints', type=int, default=SimulationParameters.qpoints, 
                        help='Number of points in q.')
    parser.add_argument('-Np', '--prpoints', type=int, default=SimulationParameters.prpoints, 
                        help='Number of points in the pair distance distribution function.')
    parser.add_argument('-N', '--Npoints', type=int, default=SimulationParameters.Npoints, 
                        help='Number of simulated points per model.')
    parser.add_argument('-expo', '--exposure', type=float, default=500, 
                        help='Exposure time in arbitrary units.')

    # Optional plot options
    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, 
                        help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-hres', '--high_res', action='store_true', default=False, 
                        help='include flag (no input) to output high resolution plot.')
    parser.add_argument('-scale', '--scale', action='store_true', default=False,
                        help='include flag (no input) to scale the simulated intensity of each model in the plots to avoid overlap.')       

    # Optional SESANS-related options (Shape2SESANS)
    parser.add_argument('-ss', '--sesans', action='store_true', default=False,
                        help='Calculate SESANS data from the SAS data.')
    parser.add_argument('-sse', '--sesans_error', type=float, default=0.02, 
                        help='Baseline SESANS error relative to max signal.')
    parser.add_argument('-Nd', '--deltapoints', type=int, default=150,
                        help='Number of points in delta.')
    
    args = parser.parse_args()

    ################################ Read input values ################################

    if args.sesans:
        # extend q-range for sesans
        Sim_par = SimulationParameters(qmin=1e-6, qmax=0.1, qpoints=20000, prpoints=args.qpoints, Npoints=args.Npoints)
    else:
        Sim_par = SimulationParameters(qmin=args.qmin, qmax=args.qmax, qpoints=args.qpoints, prpoints=args.qpoints, Npoints=args.Npoints)
    
    subunit_type = args.subunit_type
    if subunit_type is None:
        raise argparse.ArgumentError(subunit_type, "No subunit type was given as an input.")
    
    dimensions = args.dimension
    if dimensions is None:
        raise argparse.ArgumentError(dimensions, "No dimensions were given as an input.")
    for subunit, dimension in zip(subunit_type, dimensions):
         if len(subunit) != len(dimension):
            raise argparse.ArgumentTypeError("Mismatch between number subunit types (%d) and dimensions lists (%d)." % (len(subunit),len(dimension)))
         
    r_list, pr_norm_list, I_list, Isim_list, sigma_list, S_eff_list = [], [], [], [], [], [] 
    x_list, y_list, z_list, sld_list, Model_list, scale_list, name_list = [], [], [], [], [], [], []

    if args.sesans:
        delta_list,G_list,Gsim_list,sigma_G_list = [],[],[],[]

    num_models = len(subunit_type)
    if num_models == 1:
        printt(f"Simulating {num_models} model...")
    else: 
        printt(f"Simulating {num_models} models...")
    for i in range(num_models):
        
        #check model name 
        model_name = check_input(args.model_name, f"Model {i}", "model name", i)
        
        if model_name in name_list:
            model_name += '_' + str(i+1)  

        printt(" ")
        printt(f"    Generating points for Model: " + model_name)

        subunits = subunit_type[i]
        dims = dimensions[i]
        N_subunits = len(subunit_type[i])

        #read for SLD, COM, and rotation
        sld = check_3Dinput(args.sld, [1.0], "SLD", N_subunits, i)
        com = check_3Dinput(args.com, [[0, 0, 0]], "COM", N_subunits, i)
        rotation = check_3Dinput(args.rotation, [[0, 0, 0]], "rotation", N_subunits, i)

        #read exclude overlap input
        exclude_overlap = check_input(args.exclude_overlap, True, "exclude_overlap", i)

        ################################# Point cloud ################################################

        #make point cloud
        Distr = getPointDistribution(subunit_type[i],sld,dimensions[i],com,rotation,exclude_overlap,args.Npoints)

        ################################# Calculate Theoretical I(q) #################################
        printt(" ")
        printt("    Calculating intensity, I(q)...")

        #check polydispersity and concentration
        pd = check_input(args.polydispersity, 0.0, "polydispersity", i)
        conc = check_input(args.conc, 0.02, "concentration", i)

        #check structure factor parameters and default values
        Stype = check_input(args.S, 'None', "Structure type", i)
        sf_class = StructureFactor.structureFactor[Stype]
        par_dic = StructureFactor.getparname(sf_class)
        
        par = []
        
        for name in par_dic.keys():
            #attr = getattr(args, name, f"{name} could not be found. Uses default value {par_dic[name]}")
            attr = getattr(args, name, None)

            if attr is None:  
                # if not given then use default
                par.append(par_dic[name])

            elif isinstance(attr, (list, np.ndarray)):
                if len(attr) == 1:
                    # Broadcast single-element list
                    par.append(attr[0])
                elif i < len(attr):
                    # Use i-th entry
                    par.append(attr[i])
                else:
                    # if too short then fallback to default
                    par.append(par_dic[name])

            else:
                # if provided as scalar if just use it
                par.append(attr)

            #if isinstance(attr, str) or isinstance(attr, type(None)):
            #    par.append(par_dic[name])
                #printt(f"        {name} could not be found. Uses default value {par_dic[name]}.")
            #else:
            #    par.append(attr[i])


        #check interface roughness
        sigma_r = check_input(args.sigma_r, 0.0, "sigma_r", i)

        #calculate theoretical scattering
        Theo_calc = TheoreticalScatteringCalculation(System=ModelSystem(PointDistribution=Distr, 
                                                                        Stype=Stype, par=par, 
                                                                        polydispersity=pd, conc=conc, 
                                                                        sigma_r=sigma_r), 
                                                                        Calculation=Sim_par)
        Theo_I = getTheoreticalScattering(Theo_calc)

        #save models
        Model = "_".join(model_name.split())
        WeightedPairDistribution.save_pr(args.qpoints, Theo_I.r, Theo_I.pr, Model)
        StructureFactor.save_S(Theo_I.q, Theo_I.S_eff, Model)
        ITheoretical(Theo_I.q).save_I(Theo_I.I, Model)

        ######################################### Simulate I(q) ##########################################
        exposure = args.exposure
        Sim_calc = SimulateScattering(q=Theo_I.q, I0=Theo_I.I0, I=Theo_I.I, exposure=exposure)
        Sim_I = getSimulatedScattering(Sim_calc)

        # Save simulated I(q) using IExperimental
        Isim_class = IExperimental(q=Sim_I.q, I0=Theo_I.I0, I=Theo_I.I, exposure=exposure)
        Isim_class.save_Iexperimental(Sim_I.I_sim, Sim_I.I_err, Model)

        #check scaling for plot
        scale = check_input(args.scale, 1, "scale", i)

        #save data for plots
        x_list.append(np.concatenate(Distr.x))
        y_list.append(np.concatenate(Distr.y))
        z_list.append(np.concatenate(Distr.z))
        sld_list.append(np.concatenate(Distr.sld))

        r_list.append(Theo_I.r)
        pr_norm_list.append(Theo_I.pr_norm)
        I_list.append(Theo_I.I)
        S_eff_list.append(Theo_I.S_eff)

        Isim_list.append(Sim_I.I_sim)
        sigma_list.append(Sim_I.I_err)

        Model_list.append(Model)
        scale_list.append(scale)
        name_list.append(model_name)

        ######################################### SESANS ##########################################

        if args.sesans:
            # make spin echo length (delta) range (x-axis in SESANS)
            delta = np.linspace(0, 3 * np.max(Theo_I.r), args.deltapoints)
            G = calc_G_sesans(Theo_I.q,delta,Theo_I.I)

            # simulate noisy sesans data         
            G_sim,sigma_G = simulate_sesans(delta,G,args.sesans_error)
            
            # append to list (in case of multiple models)
            delta_list.append(delta)
            G_list.append(G)
            Gsim_list.append(G_sim)
            sigma_G_list.append(sigma_G)
            
    printt(" ")
    printt("Generating plots...")
    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forrestgreen']

    #plot 2D projections
    print("    2D projections: points_<model_name>.png ...")
    plot_2D(x_list, y_list, z_list, sld_list, Model_list, args.high_res, colors)
    
    #3D vizualization: generate pdb file with points
    print("    3D models: <model_name>.pdb ...")
    generate_pdb(x_list, y_list, z_list, sld_list, Model_list)
    
    #plot p(r) and I(q)
    print("    plot pr and Iq and Isim: plot.png ...")
    plot_results(Theo_I.q, r_list, pr_norm_list, I_list, Isim_list, 
                 sigma_list, S_eff_list, name_list, scale_list, args.xscale_lin, args.high_res, colors)

    #plot and save sesans
    if args.sesans:
        plot_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list, scale_list, args.high_res, colors)
        save_sesans(delta_list, G_list, Gsim_list, sigma_G_list, Model_list)

    time_total = time.time() - start_total
    printt(" ")
    printt("Simulation successfully completed.")
    printt("    Total run time: " + str(round(time_total, 1)) + " seconds.")
    printt(" ")

    f_out.close()
