"""
Utility functions for nbody simulations (mostly plotting methods).
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# nice plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def evoframe(frames, model, plot, type, nsteps, marker, logfile=None):
    """Evolve figure with model in 2D
    Evolves a figure according to model evolution. Mainly useful for matplotlib
    animations with animation.FuncAnimation.
    Args:
        frames        (int): frame number, for compat with mpl animation
        model (nbody.NBody): Nbody model to be displayed
        plot     (mpl plot): matplotlib plot showing model data (densities)
        type          (str): type of animation plot (density grid or scatter
                             plot)
        marker        (str): marker for matplotlib scatter plot, used if
                             type == 'pts' only
        logfile       (str): file where we output the energy at each step.
                             Default is None.
    """
    eg = model.evolve(nsteps=nsteps)
    egstr = 'Step {}: Total Energy is {}'.format(frames, eg)
    print(egstr)
    if logfile is not None:
        lf = open(logfile, 'a')
        lf.write(''.join([egstr, '\n']))
        lf.close()

    if type == 'grid':
        plot.set_array(model.density.T)
    elif type == 'pts':
        plot.set_data(model.pos[:, 0], model.pos[:, 1])


def evoframe3d(frames, model, plot, type, nsteps, marker, logfile=None):
    """Evolve figure with model (in 3d, collapsing model on 3rd dim)
    Evolves a figure according to model evolution. Mainly useful for matplotlib
    animations with animation.FuncAnimation.
    Args:
        frames        (int): frame number, for compat with mpl animation
        model (nbody.NBody): Nbody model to be displayed
        plot     (mpl plot): matplotlib plot showing model data (densities)
        type          (str): type of animation plot (density grid or scatter
                             plot)
        marker        (str): marker for matplotlib scatter plot, used if
                             type == 'pts' only
        logfile       (str): file where we output the energy at each step.
                             Default is None.
    """
    eg = model.evolve(nsteps=nsteps)
    egstr = 'Step {}: Total Energy is {}'.format(frames, eg)
    print(egstr)
    if logfile is not None:
        lf = open(logfile, 'a')
        lf.write(''.join([egstr, '\n']))
        lf.close()

    if type == 'grid':
        density = np.sum(model.density, axis=-1)
        plot.set_array(density.T)
    elif type == 'pts':
        plot.set_data(model.pos[:, 0], model.pos[:, 1])


def grid_animation2d(model, niter=50, show=True, savepath=None,
                     figsize=(8, 10), intv=200, title=None, repeat=False,
                     nsteps=1, style='grid', marker='o', norm=None, cmap=None,
                     logfile=None):
    """Animation of 2d nbody model
    Creates an animation of a 2d nbody model with one evolution timestep per
    frame.
    Args:
        model (nbody.NBody): Nbody model to be displayed
        niter         (int): number of evolution timesteps in the animation
        show         (bool): show the animation if true (default is true)
        savepath      (str): path where the animation should be saved. Not
                             saved when None, default is None. Must end by
                             '.gif' (file format)
        figsize     (tuple): width and heigth of mpl figure.
        intv        (float): delay (in milliseconds) between frames in
                             animation.
        title         (str): Title of the figure. No title if None (default).
        repeat       (bool): If true, repeat animation indefintely. Default is
                             False.
        nsteps        (int): Number of evolution steps for model between
                             frames.
        style         (str): 'grid' (default) or 'pts', to choose between
                             density grid or scatterplot.
        marker        (str): specifying the marker style for matplotlib scatter
                             plot (used if style=='pts' only)
        norm          (str): matplotlib grid normalization. Default is None
                             (system matplotlib settings)
        cmap          (str): specifying color map to use for density grid.
        logfile       (str): File where we output the energy at each frame.
    """
    if logfile is not None:
        lf = open(logfile, 'w')
        lf.close()

    # font parameters
    titlesize = 16

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if style == 'grid':
        plot = ax.imshow(model.density.T, norm=norm)
    elif style == 'pts':
        plot, = ax.plot(model.pos[:, 0], model.pos[:, 1], marker,
                        cmap=plt.get_cmap(cmap))
    ax.set_xlim([0, model.ngrid])
    ax.set_ylim([0, model.ngrid])
    plt.tight_layout()

    fargs = (model, plot, style, nsteps, marker, logfile)
    anim = animation.FuncAnimation(fig, evoframe, frames=niter, interval=intv,
                                   fargs=fargs, repeat=repeat)
    if savepath is not None:
        fformat = savepath.split('.')[-1]
        if fformat != 'gif':
            msg = ('Incorrect file format (.{} instead of .gif). Animation'
                   ' will not be saved.')
            warnings.warn(msg, RuntimeWarning)
        else:
            anim.save(savepath, writer='imagemagick')
    if show:
        plt.show()


def density3d(model, niter=50, show=True, savepath=None,
              figsize=(8, 10), intv=200, title=None, repeat=False,
              nsteps=1, style='grid', marker='o', norm=None, cmap=None,
              logfile=None):
    """Animation of 3d nbody model
    Creates an animation of a 3d nbody model with one evolution timestep per
    frame. The elements are summed along 3rd dim to be displayed in 2d plane.
    Args:
        model (nbody.NBody): Nbody model to be displayed
        niter         (int): number of evolution timesteps in the animation
        show         (bool): show the animation if true (default is true)
        savepath      (str): path where the animation should be saved. Not
                             saved when None, default is None. Must end by
                             '.gif' (file format)
        figsize     (tuple): width and heigth of mpl figure.
        intv        (float): delay (in milliseconds) between frames in
                             animation.
        title         (str): Title of the figure. No title if None (default).
        repeat       (bool): If true, repeat animation indefintely. Default is
                             False.
        nsteps        (int): Number of evolution steps for model between
                             frames.
        style         (str): 'grid' (default) or 'pts', to choose between
                             density grid or scatterplot.
        marker        (str): specifying the marker style for matplotlib scatter
                             plot (used if style=='pts' only)
        norm          (str): matplotlib grid normalization. Default is None
                             (system matplotlib settings)
        cmap          (str): specifying color map to use for density grid.
        logfile       (str): File where we output the energy at each frame.
    """
    if logfile is not None:
        lf = open(logfile, 'w')
        lf.close()

    # font parameters
    titlesize = 16

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if style == 'grid':
        density = np.sum(model.density, axis=-1)  # 2d collapse
        plot = ax.imshow(density.T, norm=norm, cmap=plt.get_cmap(cmap))
    elif style == 'pts':
        plot, = ax.plot(model.pos[:, 0], model.pos[:, 1], marker)
    ax.set_xlim([0, model.ngrid])
    ax.set_ylim([0, model.ngrid])
    plt.tight_layout()

    fargs = (model, plot, style, nsteps, marker, logfile)
    anim = animation.FuncAnimation(fig, evoframe3d, frames=niter,
                                   interval=intv, fargs=fargs, repeat=repeat)
    if savepath is not None:
        fformat = savepath.split('.')[-1]
        if fformat != 'gif':
            msg = ('Incorrect file format (.{} instead of .gif). Animation'
                   ' will not be saved.')
            warnings.warn(msg, RuntimeWarning)
        else:
            anim.save(savepath, writer='imagemagick')
    if show:
        plt.show()


def eplot(efile):
    """Plot energy vs step number
    Args:
        efile (str): string output from energy txt file
    """

    steps = efile.split('\n')[:-1]  # last line empty
    egy = np.array([e.split(' ')[-1] for e in steps], dtype=float)
    emin = egy.min()
    plt.plot(egy-emin)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Energy + {:.2f}'.format(-emin), fontsize=14)
    plt.title('Energy conservation', fontsize=16)
    plt.show()
