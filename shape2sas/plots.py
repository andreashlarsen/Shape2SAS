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

def plot_results(models, path, *, xscale_lin, colors):
    """
    plot results for all models (cli.ModelResult):
    - p(r)
    - calculated formfactor P(r) times structure factor S(q) on log-log or log-lin scale
    - simulated data on log-log or log-lin scale
    """
    __, ax = plt.subplots(1,3,figsize=(12,4))

    for zo, (m, color) in enumerate(zip(models, colors), start=1):
        histogram, intensity, simulated = m.points.histogram, m.intensity, m.simulated
        ax[0].plot(histogram.r, histogram.pr, zorder=zo, color=color, label='p(r), %s' % m.name)

        ax[2].errorbar(simulated.q, simulated.I, yerr=simulated.sigma, linestyle='none', marker='.', color=color,
                       label=r'$I_\mathrm{sim}(q)$, %s' % m.name, zorder=zo)

        if intensity.S[0] != 1.0 or intensity.S[-1] != 1.0:
            ax[1].plot(intensity.q, intensity.S, linestyle='--', color=color, zorder=0, label=r'$S(q)$, %s' % m.name)
            ax[1].plot(intensity.q, intensity.I, color=color, zorder=zo, label=r'$I(q)=cP(q)S(q)$, %s' % m.name)
        else:
            ax[1].plot(intensity.q, intensity.I, zorder=zo, color=color, label=r'$I(q)=cP(q)$, %s' % m.name)

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
    plt.savefig(path)
    plt.close()

def linear_fit(I, I_exp, sigma_exp):
    """
    weighted least-squares fit of the model to the data, I_exp = scale * I + background
    returns scale and background
    """

    A = np.column_stack((I, np.ones_like(I))) / sigma_exp[:, None]
    scale, background = np.linalg.lstsq(A, I_exp / sigma_exp, rcond=None)[0]

    return scale, background

def plot_fit(models, I_exp, sigma_exp, data_label, path, *, xscale_lin, colors):
    """
    plot the experimental data with the fit of each model (cli.ModelResult), I_exp = scale * I + background,
    on log-log or log-lin scale; the models are calculated at the q values of the data
    """
    N_models = len(models)
    __, ax = plt.subplots(1,N_models,figsize=(N_models*4,4), squeeze=False)

    for p, m, color in zip(ax[0], models, colors):
        q, I = m.intensity.q, m.intensity.I
        p.errorbar(q,I_exp,yerr=sigma_exp,linestyle='none',marker='.', color='grey',zorder=0, label = data_label)
        scale, background = linear_fit(I, I_exp, sigma_exp)
        p.plot(q, scale * I + background, zorder=1, color=color, label=r'%s' % m.name)

        if not xscale_lin:
            p.set_xscale('log')
        p.set_yscale('log')
        p.set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
        p.set_ylabel(r'$I(q)$ [a.u.]')
        # p.set_title('Data and fit')
        p.legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
