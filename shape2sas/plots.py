"""Plotting and 3D visualisation of the models and their scattering."""

import matplotlib.pyplot as plt
import numpy as np

def plot_projections(x, y, z, sld, path=None, *, color='blue', extent):
    """
    plot 2D projections of points onto the (x,z), (y,z) and (x,y) planes (see PointDistribution.plot_projections):
    positive contrast in color, zero contrast in grey, negative contrast in black.
    the axes go from -1.1 * extent to 1.1 * extent. saves the figure to path and closes it, or returns it if path is None
    """
    markersize = 0.5
    lim = [-1.1 * extent, 1.1 * extent]
    groups = [(sld > 0.0, color), (sld < 0.0, 'black'), (sld == 0.0, 'grey')]
    views = [(x, z, 'x', 'z', '"front"'), (y, z, 'y', 'z', '"side"'), (x, y, 'x', 'y', '"bottom"')]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (h, v, hlabel, vlabel, name) in zip(ax, views):
        for idx, c in groups:
            a.plot(h[idx], v[idx], linestyle='none', marker='.', markersize=markersize, color=c)
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_xlabel(hlabel)
        a.set_ylabel(vlabel)
        a.set_title('pointmodel, (%s,%s), %s' % (hlabel, vlabel, name))
    fig.tight_layout()

    if path is None:
        return fig
    fig.savefig(path)
    plt.close(fig)

def plot_results(q, r_list, pr_list, I_list, Isim_list, sigma_list, S_list, name_list, xscale_lin, filetype, colors):
    """
    plot results for all models:
    - p(r) 
    - calculated formfactor P(r) times structure factor S(q) on log-log or log-lin scale
    - simulated data on log-log or log-lin scale
    """
    __, ax = plt.subplots(1,3,figsize=(12,4))

    zo = 1
    for (r, pr, I, Isim, sigma, S, model_name,color) in zip (r_list, pr_list, I_list, Isim_list, sigma_list, S_list, name_list, colors):
        ax[0].plot(r,pr,zorder=zo, color=color, label='p(r), %s' % model_name)

        ax[2].errorbar(q,Isim,yerr=sigma,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % model_name,zorder=zo)

        if S[0] != 1.0 or S[-1] != 1.0:
            ax[1].plot(q, S, linestyle='--', color=color, zorder=0, label=r'$S(q)$, %s' % model_name)
            ax[1].plot(q, I,color=color, zorder=zo, label=r'$I(q)=cP(q)S(q)$, %s' % model_name)
        else:
            ax[1].plot(q, I, zorder=zo, color=color, label=r'$I(q)=cP(q)$, %s' % model_name)
        zo += 1

    ## figure settings, p(r)
    ax[0].set_xlabel(r'$r$ [$\mathrm{\AA}$]')
    ax[0].set_ylabel(r'$p(r)$')
    ax[0].set_title('pair distance distribution function')
    ax[0].legend(frameon=False)

    ## figure settings, calculated scattering
    if not xscale_lin:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[1].set_ylabel(r'$I(q)$ [cm$^{-1}$]')
    ax[1].set_title('theoretical scattering, no noise')
    ax[1].legend(frameon=False)

    ## figure settings, simulated scattering
    if not xscale_lin:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[2].set_ylabel(r'$I(q)$ [cm$^{-1}$]')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('plot.' + filetype)
    plt.close()

def linear_fit(I, I_exp, sigma_exp):
    """
    weighted least-squares fit of the model to the data, I_exp = scale * I + background
    returns scale and background
    """

    A = np.column_stack((I, np.ones_like(I))) / sigma_exp[:, None]
    scale, background = np.linalg.lstsq(A, I_exp / sigma_exp, rcond=None)[0]

    return scale, background

def plot_fit(q, I_list, I_exp, sigma_exp, name_list, data_filename, xscale_lin, filetype, colors):
    """
    plot results for all models:
    - p(r) 
    - calculated formfactor P(r) times structure factor S(q) on log-log or log-lin scale
    - simulated data on log-log or log-lin scale
    """
    N_models = len(I_list)
    width = N_models*4
    __, ax = plt.subplots(1,N_models,figsize=(width,4))

    i = 0
    for (I, model_name,color) in zip (I_list, name_list, colors):
        if N_models == 1:
            p = ax
        else:
            p = ax[i]
        p.errorbar(q,I_exp,yerr=sigma_exp,linestyle='none',marker='.', color='grey',zorder=0, label = data_filename) #label=r'$I_\mathrm{exp}(q)$')
        scale, background = linear_fit(I, I_exp, sigma_exp)
        I_model = scale * I + background
        p.plot(q, I_model, zorder=1, color=color, label=r'%s' % model_name)
        p.set_ylabel(r'$I(q)$')
        i += 1

        if not xscale_lin:
            p.set_xscale('log')
        p.set_yscale('log')
        p.set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
        p.set_ylabel(r'$I(q)$ [a.u.]')
        # p.set_title('Data and fit')
        p.legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig('fit.' + filetype)
    plt.close()
