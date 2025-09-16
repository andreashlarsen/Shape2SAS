import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compare results from Shape2SAS')
    parser.add_argument('-m', '--model_names',help='Model names')
    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, 
                            help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-r', '--high_res', action='store_true', default=False, 
                            help='include flag (no input) to output high resolution plot.')
    parser.add_argument('-s', '--scale', action='store_true', default=False,
                            help='include flag (no input) to scale the simulated intensity of each model in the plots to avoid overlap.')    
    parser.add_argument('-n', '--name', help='output filename', default='None')   

    args = parser.parse_args()

    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forrestgreen']

    models = re.split('[ ,]+', args.model_names)

    fig, ax = plt.subplots(1,3,figsize=(12,4))

    # plot p(r)
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
        plt.savefig(all_model_names)
    else:
        plt.savefig(args.name)
    plt.show()




