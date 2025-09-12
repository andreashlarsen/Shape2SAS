
import time
import argparse
import warnings
import re
from sys import argv
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from helpfunctions import (GenerateAllPoints, WeightedPairDistribution, StructureFactor, ITheoretical, IExperimental, Qsampling,plot_2D, plot_results, generate_pdb)

version = 2.1

Vectors = List[List[float]]

@dataclass
class ModelProfile:
    """Class containing parameters for
    creating a particle
    
    NOTE: Default values create a sphere with a 
    radius of 50 Å at the origin.
    """

    subunits: List[str] = field(default_factory=lambda: ['sphere'])
    p_s: List[float] = field(default_factory=lambda: [1.0]) # scattering length density
    dimensions: Vectors = field(default_factory=lambda: [[50]])
    com: Vectors = field(default_factory=lambda: [[0, 0, 0]])
    rotation_points: Vectors = field(default_factory=lambda: [[0, 0, 0]])
    rotation: Vectors = field(default_factory=lambda: [[0, 0, 0]])
    exclude_overlap: Optional[bool] = field(default_factory=lambda: True)

@dataclass
class ModelPointDistribution:
    """Point distribution of a model"""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    p: np.ndarray #scattering length density for each point
    volume_total: float

@dataclass
class SimulationParameters:
    """Class containing parameters for the simulation and default parameters"""

    qmin: float = 0.001
    qmax: float =  0.5
    qpoints: int = 400
    prpoints: int =  100
    Npoints: int = 5000
    model_name: List[str] = field(default_factory=lambda: ['Model_1'])

@dataclass
class ModelSystem:
    """Class containing parameters for
    the system"""

    PointDistribution: ModelPointDistribution
    Stype: str = field(default_factory=lambda: "None") #structure factor
    par: List[float] = field(default_factory=np.ndarray)#parameters for structure factor
    polydispersity: float = field(default_factory=lambda: 0.0)#polydispersity
    conc: float = field(default_factory=lambda: 0.02) #concentration
    sigma_r: float = field(default_factory=lambda: 0.0) #interface roughness


@dataclass
class TheoreticalScatteringCalculation:
    """Class containing parameters for simulating
    scattering for a given model system"""

    System: ModelSystem
    Calculation: SimulationParameters

@dataclass
class TheoreticalScattering:
    """Class containing parameters for
    theoretical scattering"""

    q: np.ndarray
    I0: np.ndarray
    I: np.ndarray
    S_eff: np.ndarray
    r: np.ndarray #pair distance distribution
    pr: np.ndarray #pair distance distribution
    pr_norm: np.ndarray #normalized pair distance distribution

@dataclass
class SimulateScattering:
    """Class containing parameters for
    simulating scattering"""

    q: np.ndarray = field(default_factory=np.ndarray)
    I0: np.ndarray = field(default_factory=np.ndarray)
    I: np.ndarray = field(default_factory=np.ndarray)
    exposure: Optional[float] = field(default_factory=lambda:500)

@dataclass
class SimulatedScattering:
    """Class containing parameters for
    simulated scattering"""

    I_sim: np.ndarray
    q: np.ndarray
    I_err: np.ndarray


################################ Shape2SAS functions ################################
def getPointDistribution(prof: ModelProfile, Npoints):
    """Generate points for a given model profile."""

    x_new, y_new, z_new, p_new, volume_total = GenerateAllPoints(Npoints, prof.com, prof.subunits, 
                                                  prof.dimensions, prof.rotation, 
                                                  prof.p_s, prof.exclude_overlap).onGeneratingAllPointsSeparately()
    
    return ModelPointDistribution(x=x_new, y=y_new, z=z_new, p=p_new, volume_total=volume_total)


def getTheoreticalScattering(scalc: TheoreticalScatteringCalculation) -> TheoreticalScattering:
    """Calculate theoretical scattering for a given model profile."""
    sys = scalc.System
    prof = sys.PointDistribution
    calc = scalc.Calculation
    x = np.concatenate(prof.x)
    y = np.concatenate(prof.y)
    z = np.concatenate(prof.z)
    p = np.concatenate(prof.p)

    r, pr, pr_norm = WeightedPairDistribution(x, y, z, p).calc_pr(calc.prpoints, sys.polydispersity)

    print('        calculating scattering...')
    q = Qsampling(calc.qmin, calc.qmax, calc.qpoints).onQsampling()
    I_theory = ITheoretical(q)
    I0, Pq = I_theory.calc_Pq(r, pr, sys.conc, prof.volume_total)

    S_class = StructureFactor(q, x, y, z, p, sys.Stype, sys.par)
    S_eff = S_class.getStructureFactor().structure_eff(Pq)

    I = I_theory.calc_Iq(Pq, S_eff, sys.sigma_r)

    return TheoreticalScattering(q=q, I=I, I0=I0, S_eff=S_eff, r=r, pr=pr, pr_norm=pr_norm)


def getSimulatedScattering(scalc: SimulateScattering) -> SimulatedScattering:
    """Simulate scattering for a given theoretical scattering."""

    Isim_class = IExperimental(scalc.q, scalc.I0, scalc.I, scalc.exposure)
    I_sim, I_err = Isim_class.simulate_data()

    return SimulatedScattering(I_sim=I_sim, q=scalc.q, I_err=I_err)


################################ Shape2SAS batch version ################################
if __name__ == "__main__":
    
    ################################ Define some functions ################################
    
    # file for stdout using printt function
    f_out = open('shape2sas.log','w')
    def printt(s):
        print(s)
        f_out.write('%s\n' %s)
    
    ## read input command
    input_string = 'python'
    for aa in argv:
        if ' ' in aa:
            input_string += " \"%s\"" % aa
        else:
            input_string += " %s" %aa
    
    ## welcome message
    printt('#########################################')
    printt('RUNNING shape2sas.py, version %s \nfor instructions: python shape2sas.py -h' % version)
    printt('command used: %s' % input_string)
    printt('#########################################')

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

    def separate_string(arg):

        arg = re.split('[ ,]+', arg)
        return [str(i) for i in arg]


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

    start_total = time.time()

    ################################ Define input values ################################

    #input values
    parser = argparse.ArgumentParser(description='Shape2SaS - calculates small-angle scattering from a given shape defined by the user.')
    
    #general input options
    parser.add_argument('-qmin', '--qmin', type=float, default=SimulationParameters.qmin, 
                        help='Minimum q-value for the scattering curve.')
    parser.add_argument('-qmax', '--qmax', type=float, default=SimulationParameters.qmax, 
                        help='Maximum q-value for the scattering curve.')
    parser.add_argument('-Nq', '--qpoints', type=int, default=SimulationParameters.qpoints, 
                        help='Number of points in q.')
    parser.add_argument('-expo', '--exposure', type=float, default=500, 
                        help='Exposure time in arbitrary units.')
    parser.add_argument('-Np', '--prpoints', type=int, default=SimulationParameters.prpoints, 
                        help='Number of points in the pair distance distribution function.')
    parser.add_argument('-N', '--Npoints', type=int, default=SimulationParameters.Npoints, 
                        help='Number of simulated points.')
    
    #specific input options for each model
    parser.add_argument('-modelname', '--model_name', nargs='+', action='extend',
                        help='Name of model.')
    parser.add_argument('-excluolap', '--exclude_overlap', type=str2bool, default=True, 
                        help='bool to exclude overlap.')
    parser.add_argument('-subtype', '--subunit_type', type=separate_string, nargs='+', action='extend',
                        help='Type of subunits for each model.')
    
    #--a --b --c ---> dimension 'a b c'
    parser.add_argument('-dim', '--dimension', type=float_list, nargs='+', action='append',
                        help='dimensions of subunits for each model.')
    
    parser.add_argument('-p', '--p', type=float, nargs='+', action='append',
                        help='scattering length density.')
    
    #--x --y --z ---> com =  'x y z'
    parser.add_argument('-com', '--com', type=float_list, nargs='+', action='append', 
                        help='displacement for each subunits in each model.')
    parser.add_argument('-rotation', '--rotation', type=float_list, nargs='+', action='append', 
                        help='rotation for each subunits in each model.')
    
    parser.add_argument('-poly', '--polydispersity', type=float, nargs='+', action='extend',
                        help='Polydispersity of subunits for each model.')
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
    parser.add_argument('-conc', '--conc', type=float, nargs='+', action='extend',
                        help='volume fraction concentration.')
    parser.add_argument('-sigmar', '--sigma_r', type=float, nargs='+', action='extend',
                        help='interface roughness for each model.')
    
    #plot options
    parser.add_argument('-lin', '--xscale_lin', type=str2bool, default=True, 
                        help='bool to include linear q scale.')
    parser.add_argument('-hres', '--high_res', type=bool, default=False, 
                        help='bool to include high resolution.')
    parser.add_argument('-scale', '--scale', type=int, nargs='+', action='extend',
                        help='In the plot, scale simulated intensity of each model.')       

    args = parser.parse_args()

    ################################ Read input values ################################

    #qmin = args.qmin
    #qmax = args.qmax
 
    #qpoints = args.qpoints
    #Nbins = args.prpoints
    #Npoints = args.qpoints
    #exclude_overlap = args.exclude_overlap
    #xscale_lin = args.xscale_lin
    #high_res = args.high_res
    Sim_par = SimulationParameters(qmin=args.qmin, qmax=args.qmax, qpoints=args.qpoints, prpoints=args.qpoints, Npoints=args.Npoints)

    subunit_type = args.subunit_type
    if subunit_type is None:
        raise argparse.ArgumentError(subunit_type, "No subunit type was given as an input.")
    
    dimensions = args.dimension
    if dimensions is None:
        raise argparse.ArgumentError(dimensions, "No dimensions were given as an input.")
    for subunit, dimension in zip(subunit_type, dimensions):
         if len(subunit) != len(dimension):
            raise argparse.ArgumentTypeError("Mismatch between subunit types and dimensions.")
         
    if subunit_type == 'sphere':
        print(len(dimensions[0][0]))

    r_list, pr_norm_list, I_list, Isim_list, sigma_list, S_eff_list = [], [], [], [], [], [] 
    x_list, y_list, z_list, p_list, Model_list, scale_list, name_list = [], [], [], [], [], [], []

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
        N_subunits = len(subunits)

        #check for SLD, COM, and rotation
        p_s = check_3Dinput(args.p, [1.0], "SLD", N_subunits, i)
        com = check_3Dinput(args.com, [[0, 0, 0]], "COM", N_subunits, i)
        rotation = check_3Dinput(args.rotation, [[0, 0, 0]], "rotation", N_subunits, i)

        Profile = ModelProfile(subunits=subunits, p_s=p_s, dimensions=dims, 
                     com=com, rotation_points=com, rotation=rotation, 
                     exclude_overlap=args.exclude_overlap)

        #Generate points
        Distr = getPointDistribution(Profile, args.Npoints)

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
                # Not given → use default
                par.append(par_dic[name])

            elif isinstance(attr, (list, np.ndarray)):
                if len(attr) == 1:
                    # Broadcast single-element list
                    par.append(attr[0])
                elif i < len(attr):
                    # Use i-th entry
                    par.append(attr[i])
                else:
                    # Too short → fallback to default
                    par.append(par_dic[name])

            else:
                # Provided as scalar → just use it
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
        #Model = f'{i}'
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
        p_list.append(np.concatenate(Distr.p))

        r_list.append(Theo_I.r)
        pr_norm_list.append(Theo_I.pr_norm)
        I_list.append(Theo_I.I)
        S_eff_list.append(Theo_I.S_eff)

        Isim_list.append(Sim_I.I_sim)
        sigma_list.append(Sim_I.I_err)

        Model_list.append(Model)
        scale_list.append(scale)
        name_list.append(model_name)
    
    printt(" ")
    printt("Generating plots...")
    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forrestgreen']

    #plot 2D projections
    print("    2D projections: points_<model_name>.png ...")
    plot_2D(x_list, y_list, z_list, p_list, Model_list, args.high_res, colors)
    
    #3D vizualization: generate pdb file with points
    print("    3D models: <model_name>.pdb ...")
    generate_pdb(x_list, y_list, z_list, p_list, Model_list)
    
    #plot p(r) and I(q)
    print("    plot pr and Iq and Isim: plot.png ...")
    plot_results(Theo_I.q, r_list, pr_norm_list, I_list, Isim_list, 
                 sigma_list, S_eff_list, name_list, scale_list, args.xscale_lin, args.high_res, colors)

    time_total = time.time() - start_total
    printt(" ")
    printt("Simulation successfully completed.")
    printt("    Total run time: " + str(round(time_total, 1)) + " seconds.")
    printt(" ")

    f_out.close()
