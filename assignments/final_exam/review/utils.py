"""
Utility functions for the course
"""
import camb
import numpy as np
from scipy.interpolate import griddata


def xmap(x, a, b):
    """Re-map x range to [-1, 1] intv to allow chebyshev fit

    Use transformation y=(x - 0.5 * (b+a)) / (0.5 * (b-a)) to map an arbitrary
    range to [-1, 1]. (from numerical recipes)

    Args:
        x (array): x values
    Returns:
        xnew (array): x mapped to [-1, 1]
    """
    bpa = 0.5 * (b + a)
    bma = 0.5 * (b - a)

    return (x - bpa) / bma


def get_spectrum(p, lmax_acc=2000, lmax_ret=1200):
    """Get power spectrum with CAMB
    Args:
        p (array): parameters H0, ombh2, omch2, tau, As, ns
        lmax_acc (int): maximum l index for the accuracy of calculations
        lmax_ret (int): maximum l index for the returned power spectrum
    Returns:
        tt: TT power spectrum for multipole indices 2 (quadrupole) to lmax_ret
    """
    H0 = p[0]
    ombh2 = p[1]
    omch2 = p[2]
    tau = p[3]
    As = p[4]
    ns = p[5]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=0.06,
                       tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_for_lmax(lmax_acc, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_ret)
    cmb = powers['total']
    tt = cmb[:, 0][2:]  # don't need first to indices

    return tt


def getflat(chains):
        """
        Get flat chains from MCMC chain array (for corner plot, e.g.)
        Args:
            chains (array): MCMC output chains with shape (nwalk, nsteps, ndim)
        Returns
            flatchains (array): flattened chains along walkers axis, with new
                                shape (nwalk*nsteps, ndim)
        """
        s = chains.shape
        return np.copy(chains.reshape(s[0] * s[1], s[2]))


def lowres(mat):
    """Lower resolution of matrix
    Lower the resolution of an array by a factor of 2 along each dimension.
    Take the maximum in absolute value, so that mask is conserved and BC
    is too. This method is not intended for other arrays than BC or masks.
    Args:
        mat (array): 2d array with equal even dimensions along both axes
    Returns:
        newmat (array): lower resolution matrix
    """
    # sanity check
    s = mat.shape
    assert np.unique(s).size == 1, 'mat must have equal dimensions'
    assert s[0] % 2 == 0, 'mat must have even dimensions'

    # split in 2x2 blocks and take abs max of each, then reshape
    n = s[0]
    newmat = mat.reshape(n//2, 2, -1, 2).swapaxes(1, 2).reshape(-1, 2, 2)
    absmax = np.max(np.abs(newmat), axis=(1, 2))
    truemax = np.max(newmat, axis=(1, 2))
    truemin = np.min(newmat, axis=(1, 2))
    newmat = np.where(np.equal(absmax, truemax), truemax, truemin)
    newmat = newmat.reshape(n//2, n//2)

    return newmat


def _upres(mat):
    """Increase matrix resolution
    Increase the resolution of an array by a factor of 2. This will only be
    used to initialize potential on a grid, so we don't need to care about
    precise BC/mask. We still use interpolation to make convergence faster.
    Args:
        mat (array): 2d array with equal dimensions along both axes
    Returns:
        newmat (array): array with higher resolution
    """
    # sanity check
    s = mat.shape
    assert np.unique(s).size == 1, 'mat must have equal dimensions'

    # get new interpolated grid with better resolution
    n = s[0]
    sizearr = np.linspace(0, n-1, num=n)
    xx, yy = np.meshgrid(sizearr, sizearr)
    pts = np.array([xx.ravel(), yy.ravel()]).T
    sizearr = np.linspace(0, n-1, num=2*n)
    xx, yy = np.meshgrid(sizearr, sizearr)
    ipts = np.array([xx.ravel(), yy.ravel()]).T
    newmat = griddata(pts, mat.ravel(), ipts, method='cubic')
    newmat = newmat.reshape(2*n, 2*n)

    return newmat
