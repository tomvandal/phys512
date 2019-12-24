"""
Utility functions for problem 4 of the final exam
"""
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter


def powerspec(y, window=None, smooth_sig=None):
    """Power Spectrum Density

    Obtains power spectrum density from a timeseries, with the option of
    smoothening with a Gaussian filter.

    Args:
        y           (array): y values of the time series
        window      (array): array of the window used
                             (must have same size as y)
        dt          (float): spacing between samples in time domain
        smooth_sig (scalar): standard deviation of Gaussian kernel for
                             smoothing model. Not applied if None (default).
    Returns:
        powers (array): powers corresponding to PSD of y
    """
    # sanity checks
    y = np.asarray(y)
    if window is None:
        window = np.ones(y.size)
    else:
        window = np.asarray(window)
        assert window.size == y.size, 'window should have the same size as y'

    # normalization for window
    normfac = np.sqrt(np.mean(window**2))

    # get power spectrum
    yft = np.fft.rfft(y * window) / normfac
    powers = np.abs(yft)**2

    # apply Gaussian filter to smoothen the spectrum
    if smooth_sig is not None:
        powers = gaussian_filter(powers, smooth_sig)

    return powers


def matchedfilt(data, templ, noise, window=None):
    """Matched filtering

    Obtain a matched filter from singal template and data

    Args:
        data   (array): data values
        templ  (array): signal template (should have same size as data)
        noise  (array): noise model (power spectrum)
        window (array): window for data (should have same size as data)
    Returns:
        mf (array): matched filter output
    """
    # sanity checks
    data = np.asarray(data)
    templ = np.asarray(templ)
    noise = np.asarray(noise)
    assert templ.size == data.size, ('templ should have the same size as'
                                     ' data')
    if window is None:
        window = np.ones(data.size)
    else:
        window = np.asarray(window)
        assert window.size == data.size, ('window should have the same size'
                                          ' as data')

    # get normalization for window
    normfac = np.sqrt(np.mean(window**2))

    # Normalized FT of data and template
    data_ft = np.fft.rfft(data * window) / (np.sqrt(noise)*normfac)
    templ_ft = np.fft.rfft(templ * window) / (np.sqrt(noise)*normfac)

    # get matched filter
    mf = np.fft.irfft(np.conj(templ_ft) * data_ft)
    mf = np.fft.fftshift(mf)

    return mf


def newton(fun, gradfun, x, y, pguess, yerr=None, maxit=10, cstol=1e-3,
           dptol=1e-3):
    """Use Newton's method to fit a function to noise-free data
    This methods takes a callable function as an argument and always returns
    both best fit parameters and their covariance matrix.
    Args:
        fun     (callable): function to fit with positional arguments
                            (x, parameters)
        gradfun (callable): gradient of fun with positional argumetns
                            (x, parameters)
        x          (array): x data to fit
        y          (array): y (noise-free) data to fit
        yerr       (array): error on y data
        pguess     (array): initial guess on parameters
        maxit      (float): maximum number of iterations
        cstol      (float): maximum chi-sq relative change for convergence
        dpar       (float): maximum parameter variation for convergence

    Returns:
        pars (array): optimized parameters
        cov  (array): estimate of the parameters covariance
    """
    # check args
    pguess = np.array(pguess)  # make sure numpy array
    pars = pguess.copy()
    maxit = int(maxit)
    cstol = float(cstol)
    dptol = float(dptol)
    x = np.array(x)
    y = np.array(y)
    if yerr is None:
        yerr = np.ones(y.size)

    # Perform Newton solver
    chisq_prev = 1e4  # chi2 initialization
    for j in range(maxit):

        pred = fun(x, pars)
        grad = gradfun(x, pars)

        res = (y - pred) / yerr
        chisq = np.sum(res**2)  # no ebars => no weights

        # generate matrix objects
        res = np.matrix(res).T
        grad = np.matrix(grad)

        # solve linear system
        lhs = grad.T * grad
        rhs = grad.T * res
        dpars = np.linalg.inv(lhs) * (rhs)
        dpars = np.asarray(dpars).flatten()
        for k in range(pguess.size):
            pars[k] = pars[k] + dpars[k]

        # convergence check
        csdiff = (chisq_prev - chisq) / chisq
        dprel = np.max(np.abs(dpars/pars))
        if j > 0 and csdiff < cstol and dprel < dptol:
            print("The Newton Method converged after {} iterations".format(j))
            break
        if j == maxit-1:
            msg = ("maxiter was reached without convergence... "
                   "params may be poorly constained")
            warnings.warn(msg, RuntimeWarning)

        chisq_prev = chisq

    # estimate of parameters covariance
    finpred = fun(x, pars)
    fingrad = gradfun(x, pars)
    finres = y - finpred
    invnoise = np.diag(finres**(-2))   # assume noise on each pt is residuals
    tmp = np.dot(fingrad.T, invnoise)  # compute covariance matrix
    invcov = np.dot(tmp, fingrad)
    cov = np.linalg.inv(invcov)

    return pars, cov
