"""
Utility functions to find gravitational waves in LIGO data.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

import read_ligo as rl


def load_event(eventname, datadir, events):
    """Load data for one event

    This function loads strain and template data for a given event

    Args:
        eventname (str): name of the event
        datadir   (str): relative path to directory containing data
        events    (str): object with event info (loaded from json file)
    Returns:
        time     (array): array of time values for the event
        fs         (int): sampling rate in Hz
        strains  (tuple): tuple with the two arrays of strain values (H1, L1)
        templs   (tuple): tuple with the two arrays of signal
                          templates (H1, L1)
    """
    # get event info from json file
    event = events[eventname]
    fn_H1 = event['fn_H1']              # Hanford filename
    fn_L1 = event['fn_L1']              # Livingston filename
    fn_template = event['fn_template']  # template for matched filter later
    fs = event['fs']                    # sampling rate (useful for FT stuff)

    # load data for each detector
    strain_H1, time = rl.readfile(datadir+fn_H1)
    strain_L1, _ = rl.readfile(datadir+fn_L1)
    strains = (strain_H1, strain_L1)

    # load template for each detector
    templ_H1, templ_L1 = rl.readtemp(datadir+fn_template)
    templs = (templ_H1, templ_L1)

    return time, fs, strains, templs


def powerspec(y, window=None, smooth_sig=1):
    """Power Spectrum Density

    Obtains power spectrum density from a timeseries

    Args:
        y           (array): y values of the time series
        window      (array): array of the window used
                             (must have same size as y)
        dt          (float): spacing between samples in time domain
        smooth_sig (scalar): standard deviation of Gaussian kernel for
                             smoothing model
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
    powers = gaussian_filter(powers, smooth_sig)

    return powers


def matchedfilt(strain, templ, noise, window=None):
    """Matched filtering

    Obtain a matched filter from singal template and data

    Args:
        strain (array): strain values
        templ  (array): signal template (should have same size as strain)
        noise  (array): noise model (power spectrum)
        window (array): window for data (should have same size as strain)
    Returns:
        mf (array): matched filter output
    """
    # sanity checks
    strain = np.asarray(strain)
    templ = np.asarray(templ)
    noise = np.asarray(noise)
    assert templ.size == strain.size, ('templ should have the same size as'
                                       ' strain')
    if window is None:
        window = np.ones(strain.size)
    else:
        window = np.asarray(window)
        assert window.size == strain.size, ('window should have the same size'
                                            ' as strain')

    # get normalization for window
    normfac = np.sqrt(np.mean(window**2))

    # Normalized FT of data and template
    strain_ft = np.fft.rfft(strain * window) / (np.sqrt(noise)*normfac)
    templ_ft = np.fft.rfft(templ * window) / (np.sqrt(noise)*normfac)

    # get matched filter
    mf = np.fft.irfft(np.conj(templ_ft) * strain_ft)
    mf = np.fft.fftshift(mf)

    return mf


def get_snr(mf, templ, noise, window=None):
    """Signal to noise ratio

    Get SNR from matched filter output and signal template

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


def expect_snr(templ, noise, window=None):
    """Expected Analytic SNR

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


def get_hf(freqs, templ, noise, window):
    """Half weight frequency

    Get frequency where half the weight are above and half are below

    Args:
        freqs  (array): frequency values for FT of template and noise PSD
        templ  (array): signal template
        noise  (array): noise model (power spectrum)
        window (array): window values
    Returns:
        hf (float): half weight frequency
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
    hf = freqs[half_ind]

    return hf


def gauss(x, a, mu, s):
    """Gaussian function

    Non-normalized Gaussian

    Args:
        x  (array): x input values
        a  (float): amplitude
        mu (float): mean value of x
        s  (float): std. dev. in x
    Returns:
        g (array): y value of the Gaussian at each x
    """
    g = a * np.exp(-0.5*(x-mu)**2/s**2)

    return g


def toa(time, snr, sguess=0.001, nside=5):
    """Error in Time of Arrival

    Get error in time of arrival from SNR and time values

    Args:
        time (array): time values of the data
        snr  (array): SNR of data given signal template
        nside  (int): number of points to keep on each side of the max
    Returns:
        eta (float): uncertainty in time of arrival (1 sigma)
    """
    # sanity checks
    time = np.asarray(time)
    snr = np.asarray(snr)
    assert time.size == snr.size, "time and snr should have the same size"

    # get guess of each parameter for the (non-normalized) Gaussian
    a0 = np.max(snr)  # fixed there
    imax = np.argmax(snr)
    ta = time[imax]  # time of arrival, fixed
    s0 = sguess  # free param

    # get best parameter values (error in time of arrival)
    eta, _ = curve_fit(lambda x, s: gauss(x, a0, ta, s),
                       time[imax-nside:imax+nside], snr[imax-nside:imax+nside],
                       p0=[s0])

    return ta, eta.item()
