"""Plotting and saving of SESANS results."""

import matplotlib.pyplot as plt
import numpy as np

def plot_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list, filetype, colors):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    zo = 1
    for (d, G, Gsim, sigmaG, model_name, color) in zip (delta_list, G_list, Gsim_list, sigma_G_list, name_list, colors):

        ax[0].plot(d, G, zorder=zo, color=color,label=r'$G$, %s' % model_name)
        ax[0].set_ylabel(r'$G(\delta)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[0].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[0].set_title('theoretical SESANS, no noise')
        ax[0].legend(frameon=False)
                         

        ax[1].errorbar(d,Gsim,yerr=sigmaG,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % model_name,zorder=zo)
        ax[1].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[1].set_ylabel(r'$\ln(P)/(t\lambda^2)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[1].set_title('simulated SESANS, with noise')
        ax[1].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('sesans.' + filetype)
    plt.close()

def save_sesans(delta_list, G_list, Gsim_list, sigma_G_list, name_list):

    for (d, G, Gsim, sigmaG, model_name) in zip (delta_list, G_list, Gsim_list, sigma_G_list, name_list):
         
        with open('%s/G_%s.ses' % (model_name,model_name),'w') as f:
            f.write('# Theoretical SESANS data\n')
            f.write('# %-12s %-12s\n' % ('delta','G'))
            for i in range(len(d)):
                f.write('  %-12.5e %-12.5e\n' % (d[i], G[i]))
        
        with open('%s/lnP_%s.ses' % (model_name,model_name),'w') as f:
            #f.write('# Simulated SESANS data, with noise\n')
            #f.write('# %-12s %-12s %-12s\n' % ('delta','G','sigma_G'))
            f.write('FileFormatVersion      1.0\n')
            f.write('DataFileTitle          Shape2SAS Simulated SESANS data\n')
            f.write('Sample                 Simulated particle\n')
            f.write('Thickness              1.310000\n')
            f.write('Thickness_unit         mm\n')
            f.write('Theta_zmax             0.10000000000000001\n')
            f.write('Theta_zmax_unit        radians\n')
            f.write('Theta_ymax             0.10000000000000001\n')
            f.write('Theta_ymax_unit        radians\n')
            f.write('SpinEchoLength_unit    A\n')
            f.write('Depolarisation_unit    A-2 cm-1\n')
            f.write('Wavelength_unit        A\n')
            f.write('\n')
            f.write('BEGIN_DATA\n')
            f.write('SpinEchoLength Depolarisation Depolarisation_error Wavelength\n')
            for i in range(1,len(d)):
                f.write('  %-12.5e %-12.5e %-12.5e 2\n' % (d[i], Gsim[i], sigmaG[i]))

