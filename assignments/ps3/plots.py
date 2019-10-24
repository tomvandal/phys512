"""
Plotting methods used in this assignment
"""
import numpy as np
import matplotlib.pyplot as plt
import corner

import opt_methods as om


# matplotlib settings (looks nicer)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def spectrum(ind, power, epower=None, ind_fit=None, power_fit=None,
             fit_label="Fit", show=True, savepath=None):
    """Plot of CMB power spectrum
    Args:
        ind: multipole indices of data
        power: power data
        epower: error on power (default none)
        ind_fit: multipole indices of fit
        power_fit: power fit values
        show: show the plot
        savepath: where to save the file (nowhere if None)
    """
    plt.errorbar(ind, power, yerr=epower, fmt='k.', capsize=2, alpha=0.2,
                 zorder=0, label="WMAP data")
    if power_fit is not None and ind_fit is not None:
        plt.plot(ind_fit, power_fit, 'r-', label=fit_label, zorder=1)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


def mcmc_chains(chains, names, savepath=None, show=True, title=None, tsize=22,
                scale=None, lim=True):
    if scale is None:
        scale = "linear"
    ndim = chains.shape[2]
    if ndim > 1:
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels = names
        for i in range(ndim):
            ax = axes[i]
            ax.plot(chains[:, :, i].T, 'k')
            if lim:
                ax.set_xlim(0, len(chains[:, :, i].T))
            ax.set_ylabel(labels[i])
            ax.set_xscale(scale)
            ax.set_yscale(scale)

        axes[-1].set_xlabel("Step number")
        if title is not None:
            fig.suptitle(title, fontsize=tsize)
    else:
        fig = plt.figure(figsize=(10, 7))
        label = names[0]
        plt.plot(chains[:, :, 0].T, 'k', alpha=0.3)
        # plt.xlim(0, len(chains[:, :, 0].T))
        plt.ylabel(label)
        plt.xlabel("Step number")
        if title is not None:
            plt.title(title, fontsize=tsize)
    if savepath == "pdf":
        return fig
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)


def mcmc_corner(samples, labels, show_titles=True, title=None, tsize=22,
                truths=None, ticksize=12, label_kwargs={"fontsize": 20},
                savepath=None, show=True, title_kwargs=None, figsize=None):

    if len(samples.shape) == 3:  # make sure chain is flat
        samples = om.getflat(samples)
    ndim = samples.shape[1]

    figure = corner.corner(samples, labels=labels, show_titles=show_titles,
                           truths=truths, quantiles=[0.5],
                           label_kwargs=label_kwargs,
                           title_kwargs=title_kwargs)
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
