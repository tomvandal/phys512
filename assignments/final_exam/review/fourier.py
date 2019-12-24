"""
Methods for Fourier analysis
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from . import functions as fun


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


def get_snr(mf, templ, noise, window=None):
    """Signal to noise ratio

    Get SNR from matched filter output and signal template.

    Args:
        mf     (array): Matched filter output
        templ  (array): signal template
        noise  (array): noise model (power spectrum)
        window (array): window values
    Returns:
        snr (array): signal to noise ratio
    """
    # sanity checks
    mf = np.asarray(mf)
    templ = np.asarray(templ)
    noise = np.asarray(noise)
    if window is None:
        window = np.ones(templ.size)
    else:
        window = np.asarray(window)
        assert window.size == templ.size, ('window should have the same size'
                                           ' as templ')

    # get normalization
    normfac = np.sqrt(np.mean(window**2))

    # normalized FT of template
    templ_ft = np.fft.rfft(templ * window) / normfac

    # get snr
    snr_rt = np.sqrt(np.conj(templ_ft) * templ_ft / noise)
    snr = np.abs(mf * np.fft.fftshift(np.fft.irfft(snr_rt)))

    return snr


def analytic_snr(templ, noise, window=None):
    """Analytic SNR

    Get Expected SNR from signal template and noise model

    Args:
        templ  (array): signal template
        noise  (array): noise model (power spectrum)
        window (array): window values
    Returns:
        snr (array): expected analytic SNR in time domain
    """
    # sanity checks
    templ = np.asarray(templ)
    noise = np.asarray(noise)
    if window is None:
        window = np.ones(templ.size)
    else:
        window = np.asarray(window)
        assert window.size == templ.size, ('window should have the same size'
                                           ' as templ')

    # get normalization
    normfac = np.sqrt(np.mean(window**2))

    # normalized FT of template
    templ_ft = np.fft.rfft(templ * window) / normfac

    # get analytic SNR
    snr = np.abs(np.fft.irfft(templ_ft / np.sqrt(noise)))

    return snr


def get_hwf(freqs, templ, noise, window):
    """Half weight frequency

    Get frequency where half the weight are above and half are below

    Args:
        freqs  (array): frequency values for FT of template and noise PSD
        templ  (array): signal template
        noise  (array): noise model (power spectrum)
        window (array): window values
    Returns:
        hwf (float): half weight frequency
    """
    # sanity checks
    freqs = np.asarray(freqs)
    templ = np.asarray(templ)
    noise = np.asarray(noise)
    assert freqs.size == noise.size, ('freqs should have the same size as'
                                      ' noise')
    if window is None:
        window = np.ones(templ.size)
    else:
        window = np.asarray(window)
        assert window.size == templ.size, ('window should have the same size'
                                           ' as templ')

    # get normalization
    normfac = np.sqrt(np.mean(window**2))

    # normalized FT of template
    templ_ft = np.fft.rfft(templ * window) / normfac

    # get cumulative sum of template power spectrum
    # (weight at each discrete frequency)
    power_weight = np.cumsum(np.abs(templ_ft / np.sqrt(noise))**2)

    # find index where the value is the closest to half of the total
    # (max weight)
    half_ind = np.argmin(np.abs(power_weight - 0.5*np.max(power_weight)))

    # find corresponding frequency
    hwf = freqs[half_ind]

    return hwf


def snr_profile(time, snr, sguess=0.001, nside=5):
    """Fit gaussian profile to SNR

    Get peak center and width from SNR Gaussian profile.

    Args:
        time (array): time values of the data
        snr  (array): SNR in time-domain
        nside  (int): number of points to keep on each side of the max from the
                      entire dataset.
    Returns:
        tmax (float): peak center
        etmax (float): uncertainty in time at center of peak
    """
    # sanity checks
    time = np.asarray(time)
    snr = np.asarray(snr)
    assert time.size == snr.size, "time and snr should have the same size"

    # get guess of each parameter for the (non-normalized) Gaussian
    a0 = np.max(snr)  # fixed there
    imax = np.argmax(snr)
    tmax = time[imax]  # time of arrival, fixed
    s0 = sguess  # free param

    # get best parameter values (error in time of arrival)
    etmax, _ = curve_fit(lambda x, s: fun.gauss(x, a0, tmax, s),
                         time[imax-nside:imax+nside],
                         snr[imax-nside:imax+nside], p0=[s0])

    return tmax, etmax
