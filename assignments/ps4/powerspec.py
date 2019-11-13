"""
Useful function to obtain and manipulate power spectrum decomposition
"""
import numpy as np
from scipy.fftpack import rfft


def powerspec(y, fs=1.0, winfun=None, nfft=256, noverlap=None):
    """Power spectrum of a dataset

    Get power spectrum with Welch's average periodogram method.
    (I know of at least two implementations of this, one in scipy and one in
    matplotlib.mlab, but let's implement our own for this assignment)

    Args:
        y:        y values of the time series. (should be real numbers)
        fs:       sampling frequency of the timeseries in Hz.
        winfun:   window function to use. This should be a function taking one
                  argument (n).
        nfft:     number of points in each segment of the FFT This should be
                  an even number (this function is limited to that case).
        noverlap: number of points overlapping between segments.
                  Default (None) is nfft/2
    Returns:
        powers: power values of the spectrum
        freqs:  frequency corresponding to each power value
    """
    # sanity checks
    y = np.asarray(y)
    assert y.ndim == 1, "y should be a 1-D array."
    assert nfft < y.size, "nfft should be less than size of y."
    assert nfft % 2 == 0, "nfft should be an even number."
    if noverlap is None:
        noverlap = nfft // 2  # default value
    elif noverlap >= nfft:
        raise ValueError('noverlap should be less than nfft.')

    # get window and normalization
    if winfun is not None:
        assert callable(winfun), ("Window should be a callable object taking"
                                  "one argument.")
        win = winfun(nfft)
        normfac = 1.0 / np.sum(win)**2  # rescaling for a power spectrum
    else:
        win = 1.0
        normfac = 1.0

    # get step size and boundary indices
    stepsize = nfft - noverlap
    inds = np.arange(0, y.size-nfft+1, stepsize)

    # loop through segments and get powers
    powers = np.empty(nfft // 2 + 1)  # for even number of pts
    for i, ind in enumerate(inds):
        yi = y[ind:ind+nfft]  # y for current segment
        # yft = np.fft.rfft(yi*win, nfft)  # real FT of segment
        yft = rfft(yi*win, nfft)

        # Add segment to power spectrum and average at the same time.
        # There is twice as many points in the FT compared to power spectrum,
        # so we sum the squares of subsequent points (::2 indexing below)
        if i == 0:
            powers[[0, -1]] = yft[[0, -1]]**2             # handle boundaries
            print(powers[1:-1].size, yft[1:-1:2].size, yft[2::2].size)
            powers[1:-1] = yft[1:-1:2]**2 + yft[2::2]**2  # middle
        else:
            powers *= i / (i+1.0)  # rescale previous steps for avg
            powers[[0, -1]] += yft[[0, -1]]**2 / (i+1.0)
            powers[1:-1] += (yft[1:-1:2]**2 + yft[2::2]**2) / (i+1.0)

    # normalize: twice as big for middle points because they result from a sum
    powers[1:-1] *= 2*normfac
    powers[[0, -1]] *= normfac

    # frequencies: fs gives the spacing and nfft rescales by number of points
    # (since nfft is not the same as the size of powers)
    freqs = np.arange(powers.size) * (fs / nfft)

    return powers, freqs


def blackwin(nmax):
    """Blackman window function
    nmax: length of the window
    """
    n = np.arange(nmax)
    return 0.42 - 0.5*np.cos(2*np.pi*n/nmax) + 0.08 * np.cos(4*np.pi*n/nmax)


def sinwin(nmax):
    """Sine window function
    nmax: length of the window
    """
    n = np.arange(nmax)
    return np.sin(np.pi*n/nmax)
