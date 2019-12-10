"""
Utility functions to use for nbody simulations (mostly plotting methods).
"""
import warnings

import matplotlib.pyplot as plt
from matplotlib import animation

# nice plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def evoframe(frames, model, plot, type, nsteps, marker):
    """Evolve figure with model
    Evolves a figure according to model evolution. Mainly useful for matplotlib
    animations with animation.FuncAnimation.
    Args:
        frames (int): frame number, for compat with mpl animation
        model (nbody.NBody): Nbody model to be displayed
        plot (mpl plot): matplotlib plot showing model data (densities)
        type (str): type of animation plot
    """
    eg = model.evolve(nsteps=nsteps)
    print(eg)
    if type == 'grid':
        plot.set_array(model.density.T)
    elif type == 'pts':
        plot.set_data(model.pos[:, 0], model.pos[:, 1])


def grid_animation2d(model, niter=50, show=True, savepath=None,
                     figsize=(8, 10), intv=200, ret_fig=False, ret_ani=False,
                     title=None, repeat=False, nsteps=1, style='grid',
                     marker='o', norm=None, cmap=None):
    """Animation of 2d nbody model
    Creates an animation of a 2d nbody model with one evolution timestep per
    frame.
    Args:
        model (nbody.NBody): Nbody model to be displayed
        niter (int): number of evolution timesteps in the animation
        show (bool): show the animation if true (default is true)
        savepath (str): path where the animation should be saved. Not saved
                        when None, default is None.
                        Must end by '.gif' (file format)
        figsize (tuple): width and heigth of mpl figure.
        intv (float):  delay (in milliseconds) between frames in animation.
        ret_fig (bool): return figure used in animation if true, default is
                        false.
        ret_ani (bool): return animation object if true. Default is false.
        title (str): Title of the figure. No title if None (default).
    Returns:
        fig (matplotlib figure): figure used in the animation, returned only
                                 if ret_fig is true.
        anim (matplotlib animation): animation used in the function, returned
                                     only if ret_ani is true
    """
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

    anim = animation.FuncAnimation(fig, evoframe, frames=niter, interval=intv,
                                   fargs=(model, plot, style, nsteps, marker),
                                   repeat=repeat)
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
