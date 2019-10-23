import numpy as np
import matplotlib.pyplot as plt
import corner

from gprv import plots
from gprv import utils as ut


def mcmc_corner(samples, labels, show_titles=True, title=None, tsize=22,
                truths=None, ticksize=12, label_kwargs={"fontsize": 20},
                savepath=None, show=True, title_kwargs=None, figsize=None):

    if len(samples.shape) == 3:  # make sure chain is flat
        samples = ut.getflat(samples)
    ndim = samples.shape[1]

    figure = corner.corner(samples, labels=labels, show_titles=show_titles,
                           truths=truths, quantiles=[0.5],
                           label_kwargs=label_kwargs,
                           title_kwargs=title_kwargs,
                           plot_contours=False,
                           bins=20)
    if figsize is not None:
        figure.set_size_inches(figsize)

    quant16 = np.percentile(samples, 16, axis=0)
    quant84 = np.percentile(samples, 84, axis=0)

    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.axvspan(quant16[i], quant84[i], color='k', alpha=0.19)

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
    if title is not None:
        figure.suptitle(title, fontsize=tsize)
    if savepath == "pdf":
        return figure
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close(figure)


# load chains
chains = np.loadtxt("chainsprog.dat")

print(chains.shape)

pnames = [r"$H_0$", r"$\omega_b h^2$", r"$\omega_c h^2$", r"$\tau$",
          r"$A_s$", r"$n_s$"]
plots.mcmc_chains(np.array([chains]),
                  pnames, savepath=None,
                  show=True, title=None, tsize=22)
mcmc_corner(chains, pnames, show_titles=True, title=None, tsize=22,
                  truths=None, ticksize=12, label_kwargs={"fontsize": 20},
                  savepath=None, show=True, title_kwargs=None, figsize=None)
