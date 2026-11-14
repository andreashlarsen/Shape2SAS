"""Plotting of SESANS results."""

import matplotlib.pyplot as plt

def plot_sesans(models, path, *, colors):
    """plot theoretical and simulated SESANS data of all models (cli.ModelResult)"""
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    for zo, (m, color) in enumerate(zip(models, colors), start=1):
        sesans, simulated = m.sesans, m.simulated_sesans
        ax[0].plot(sesans.delta, sesans.G, zorder=zo, color=color,label=r'$G$, %s' % m.name)
        ax[0].set_ylabel(r'$G(\delta)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[0].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[0].set_title('theoretical SESANS, no noise')
        ax[0].legend(frameon=False)

        ax[1].errorbar(simulated.delta,simulated.G,yerr=simulated.sigma,linestyle='none',marker='.', color=color,label=r'$I_\mathrm{sim}(q)$, %s' % m.name,zorder=zo)
        ax[1].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
        ax[1].set_ylabel(r'$\ln(P)/(t\lambda^2)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
        ax[1].set_title('simulated SESANS, with noise')
        ax[1].legend(frameon=True)

    ## figure settings
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
