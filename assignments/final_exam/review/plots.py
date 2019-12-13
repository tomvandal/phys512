"""
Plotting methods
"""
import numpy as np
import matplotlib.pyplot as plt
import corner

import utils as ut


# matplotlib settings (looks nicer)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def mcmc_chains(chains, names, savepath=None, show=True, title=None, tsize=22,
                scale="linear", lim=True):
    """Chains by parameter for MCMC
    Args:
        chains     (array): array with MCMC sample chains, with shape
                            (nwalk, nsteps, ndim)
        names (array_like): name of each parameters, must have length ndim
        savepath     (str): path of file where plot will be saved
        show        (bool): show plot if true (default)
        title        (str): Plot title
        scale        (str): matplotlib scale. Main options are 'linear', or
                            'log'.
        lim          (true): limit x-axis to sample chain ranges
        """

    # generate plot
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

    # save and/or show
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)


def mcmc_corner(samples, labels, show_titles=True, title=None, tsize=22,
                truths=None, ticksize=12, label_kwargs={"fontsize": 20},
                savepath=None, show=True, title_kwargs=None, figsize=None,
                contours=True):
    """Corner plot of MCMC samples
    Args:
        samples     (array): array with MCMC flattened chains, with shape
                             (nwalk*nsteps, ndim)
        labels (array_like): name of each parameters, must have length ndim
        show_titles  (bool): show parameter value on top of 1d hists if true,
                             default is true
        title         (str): General plot title
        tsize         (int): title fontsize
        truths (array_like): true values to show as solid lines on corner plot
        ticksize      (int): fontsize for tick labels
        label_kwargs (dict): kwargs (like font) for axis labels. Default is
                             {"fontsize": 20}
        savepath      (str): path of file where plot will be saved
        show         (bool): show plot if true (default)
        title_kwargs (dict): kwargs (like font) for 1d histo titles. Default is
                             None.
        figssize    (tuple): two integers giving figure dimensions. Default is
                             None, for corner
        contours     (bool): Show delimited contours on samples if true
                             (default is true).
    """
    if len(samples.shape) == 3:  # make sure chain is flat
        samples = ut.getflat(samples)
    ndim = samples.shape[1]

    figure = corner.corner(samples, labels=labels, show_titles=show_titles,
                           truths=truths, quantiles=[0.5],
                           plot_contours=contours, label_kwargs=label_kwargs,
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

    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close(figure)
