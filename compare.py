import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # input arguments
    parser = argparse.ArgumentParser(description='Compare results from Shape2SAS')
    parser.add_argument('-m', '--model_names',help='Model names')
    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, 
                            help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-hres', '--high_res', action='store_true', default=False, 
                            help='include flag (no input) to output high resolution plot.')
    parser.add_argument('-s', '--scale', action='store_true', default=False,
                            help='include flag (no input) to scale the simulated intensity of each model in the plots to avoid overlap.')    
    parser.add_argument('-n', '--name', help='output filename', default='None')   
    parser.add_argument('-g', '--grid',action='store_true',help="add grid in 2D point representation",default=False)
    args = parser.parse_args()

    # colors and models
    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forrestgreen']
    models = re.split('[ ,]+', args.model_names)

    # plot data: p(r), I(q), Isim(q)
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    scale_factor = 1
    zo=1
    all_model_names = ''
    for i,model in enumerate(models):
        pr_filename = 'pr_' + model + '.dat'
        r,pr = np.genfromtxt(pr_filename,skip_header=1,unpack=True)
        ax[0].plot(r,pr,color=colors[i],label=model)

        Iq_filename = 'Iq_' + model + '.dat'
        q,I = np.genfromtxt(Iq_filename,skip_header=1,unpack=True)
        ax[1].plot(q,I,color=colors[i],label=model)

        Isim_filename = 'Isim_' + model + '.dat'
        q,Isim,sigma = np.genfromtxt(Isim_filename,skip_header=1,unpack=True)
        if args.scale: 
            ax[2].errorbar(q,Isim*scale_factor,yerr=sigma*scale_factor,linestyle='none',marker='.', color=colors[i],label=r'$I_\mathrm{sim}(q)$, %s, scaled by %1.0e' % (model,scale_factor),zorder=1/zo)
            scale_factor *= 0.1
        else:
            ax[2].errorbar(q,Isim,yerr=sigma,linestyle='none',marker='.', color=colors[i],label=r'$I_\mathrm{sim}(q)$, %s' % model,zorder=zo)
        if i > 0:
            all_model_names += '_'
        all_model_names += model

    ax[0].set_xlabel(r'$r$ [$\mathrm{\AA}$]')
    ax[0].set_ylabel(r'$p(r)$')
    ax[0].set_title('pair distance distribution function')
    ax[0].legend(frameon=False)

    if not args.xscale_lin:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[1].set_ylabel(r'normalized $I(q)$')
    ax[1].set_title('normalized scattering, no noise')
    ax[1].legend(frameon=False)

    if not args.xscale_lin:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[2].set_ylabel(r'$I(q)$ [a.u.]')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].legend(frameon=True)

    plt.tight_layout()
    if args.name == 'None':
        plt.savefig(all_model_names + '_data')
    else:
        plt.savefig(args.name + '_data')

    # plot points: 2D projection
    fig, ax = plt.subplots(len(models),3,figsize=(12,4*len(models)))
    #f,   ax = plt.subplots(1,3,figsize=(12,4))
    #plt.show()

    markersize = 0.5

    # find max dimension
    max_l = 0
    for i,model in enumerate(models):
        points_filename = 'points_' + model + '.txt'
        x,y,z,sld = np.genfromtxt(points_filename,skip_header=1,unpack=True)
        if np.amax(abs(x)) > max_l:
            max_l = np.amax(abs(x))
        if np.amax(abs(y)) > max_l:
            max_l = np.amax(abs(y))
        if np.amax(abs(z)) > max_l:
            max_l = np.amax(abs(z)) 
        max_l *= 1.1
         
    lim = [-max_l, max_l]

    for i,model in enumerate(models):

        points_filename = 'points_' + model + '.txt'
        x,y,z,sld = np.genfromtxt(points_filename,skip_header=1,unpack=True)
        
        ## find indices of positive, zero and negatative contrast
        idx_neg = np.where(sld < 0.0)
        idx_pos = np.where(sld > 0.0)
        idx_nul = np.where(sld == 0.0)        
        
        ## plot, perspective 1
        ax[i,0].plot(x[idx_pos], z[idx_pos], linestyle='none', marker='.', markersize=markersize, color=colors[i])
        ax[i,0].plot(x[idx_neg], z[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[i,0].plot(x[idx_nul], z[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')
        ax[i,0].set_xlim(lim)
        ax[i,0].set_ylim(lim)
        ax[i,0].set_xlabel('x')
        ax[i,0].set_ylabel('z')
        if i == 0:
            ax[i,0].set_title('pointmodel, (x,z), "front"')
        if args.grid:
            ax[i,0].grid()

        ## plot, perspective 2
        ax[i,1].plot(y[idx_pos], z[idx_pos], linestyle='none', marker='.', markersize=markersize, color=colors[i]) 
        ax[i,1].plot(y[idx_neg], z[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[i,1].plot(y[idx_nul], z[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')
        ax[i,1].set_xlim(lim)
        ax[i,1].set_ylim(lim)
        ax[i,1].set_xlabel('y')
        ax[i,1].set_ylabel('z')
        if i == 0:
            ax[i,1].set_title('pointmodel, (y,z), "side"')
        if args.grid:
            ax[i,1].grid()

        ## plot, perspective 3
        ax[i,2].plot(x[idx_pos], y[idx_pos], linestyle='none', marker='.', markersize=markersize, color=colors[i]) 
        ax[i,2].plot(x[idx_neg], y[idx_neg], linestyle='none', marker='.', markersize=markersize, color='black')
        ax[i,2].plot(x[idx_nul], y[idx_nul], linestyle='none', marker='.', markersize=markersize, color='grey')    
        ax[i,2].set_xlim(lim)
        ax[i,2].set_ylim(lim)
        ax[i,2].set_xlabel('x')
        ax[i,2].set_ylabel('y')
        if i == 0:
            ax[i,2].set_title('pointmodel, (x,y), "bottom"')
        if args.grid:
            ax[i,2].grid()
    
    plt.tight_layout()
    if args.name == 'None':
        plt.savefig(all_model_names + '_points')
    else:
        plt.savefig(args.name + '_points')
    plt.show()




