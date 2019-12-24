"""
Function containing a script to analyze LIGO data using methods from utils.py.
Nothing is returned by the function. The results printed and plotted.
No save option is implemented for the plots since this function is meant
to be used in an ipynb file.
"""
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
from astropy import units as u

# make nicer plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def find_gw(events, eventname, datadir):
    """Find Gravitational waves

    Use utils.py function to search for GWs in LIGO data for a given event

    Args:
        events    (dict): metadata about events, loaded from json file
        eventname (str): name of a specific event
        datadir   (str): relative path to where the data is stored
    """

    # permanent parameters (for plotting)
    titlesize = 20
    labelsize = 14
    legsize = 14

    # ############# OVERVIEW OF DATA #############
    print('DATA OVERVIEW')
    # load everything
    time, fs, strains, templs = ut.load_event(eventname, datadir, events)
    strain_H1, strain_L1 = strains
    templ_H1, templ_L1 = templs
    toff = time.min()  # set time offset

    # get frequencies for the dataset
    freqs = np.fft.rfftfreq(strain_H1.size, 1.0/fs)

    # first plot of data and templates
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    ax[0].plot(time-toff, strain_H1*1e19, linewidth=0.5, color='b',
               label='H1 Data')
    ax[0].plot(time-toff, strain_L1*1e19, linewidth=0.5, color='r',
               label='L1 Data')
    ax[0].set_ylabel(r'Strain $\times 10^{19}$', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('LIGO Data for event {}'.format(eventname),
                    fontsize=titlesize)
    ax[1].plot(time-toff, templ_H1*1e19, linewidth=0.5, color='b',
               label='H1 Template')
    ax[1].plot(time-toff, templ_L1*1e19, linewidth=0.5, color='r',
               label='L1 Template')
    ax[1].set_ylabel(r'Strain $\times 10^{19}$', fontsize=labelsize)
    ax[1].set_xlabel(r'GPS Time-{} $\times 10^{{9}}$ s'.format(toff/1e9),
                     fontsize=labelsize)
    ax[1].legend(loc=1, fontsize=legsize)
    plt.show()
    print()
    print()

    # ############# PART A (NOISE MODEL) #############
    print('(a) NOISE MODEL')
    # get power spectrum for each detector
    window = np.blackman(strain_H1.size)
    powers_H1 = ut.powerspec(strain_H1, window=window)
    powers_L1 = ut.powerspec(strain_L1, window=window)

    # plot the ASD
    plt.loglog(freqs, np.sqrt(powers_H1), 'b', label='H1')
    plt.loglog(freqs, np.sqrt(powers_L1), 'r', label='L1')
    plt.xlim(20, 2000)  # focus on interesting range
    plt.ylabel(r'ASD (strain/$\sqrt{\textrm{Hz}}$)', fontsize=labelsize)
    plt.xlabel(r'Frequency (Hz)', fontsize=labelsize)
    plt.title(
        r'Log-log plot of the Amplitude Spectrums for {}'.format(eventname),
        fontsize=titlesize)
    plt.legend(loc=1, fontsize=legsize)
    plt.show()
    print()
    print()

    # ############# PART B (MATCHED FILTER) #############
    print('(b) MATCHED FILTER')
    # get filter output
    mf_H1 = ut.matchedfilt(strain_H1, templ_H1, powers_H1, window=window)
    mf_L1 = ut.matchedfilt(strain_L1, templ_L1, powers_L1, window=window)

    # show filter output
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    ax[0].plot(time-toff, mf_H1, linewidth=0.5, color='b',
               label='H1 Output')
    ax[0].set_ylabel(r'Filter Output', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('Matched Filtering Outputs in Time Domain',
                    fontsize=titlesize)
    ax[1].plot(time-toff, mf_L1, linewidth=0.5, color='r',
               label='L1 Output')
    ax[1].set_ylabel(r'Filter Output', fontsize=labelsize)
    ax[1].set_xlabel(r'GPS Time-{} $\times 10^{{9}}$ s'.format(toff/1e9),
                     fontsize=labelsize)
    ax[1].legend(loc=1, fontsize=legsize)
    plt.show()
    print()
    print()

    # ############# PART C (SNR) #############
    print('(c) SNR')
    # get snr for each detector
    snr_H1 = ut.get_snr(mf_H1, templ_H1, powers_H1, window=window)
    snr_L1 = ut.get_snr(mf_L1, templ_L1, powers_L1, window=window)
    snr_tot = np.sqrt(snr_H1**2 + snr_L1**2)

    # show SNR
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 9))
    ax[0].plot(time-toff, snr_H1, linewidth=0.5, color='b',
               label='H1 SNR')
    ax[0].set_ylabel(r'SNR', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('Signal to Noise Ratio (SNR) in Time Domain',
                    fontsize=titlesize)
    ax[1].plot(time-toff, snr_L1, linewidth=0.5, color='r',
               label='L1 SNR')
    ax[1].set_ylabel(r'SNR', fontsize=labelsize)
    ax[1].legend(loc=1, fontsize=legsize)
    ax[2].plot(time-toff, snr_tot, linewidth=0.5, color='g',
               label='Combined SNR')
    ax[2].set_ylabel(r'SNR', fontsize=labelsize)
    ax[2].set_xlabel(r'GPS Time-{} $\times 10^{{9}}$ s'.format(toff/1e9),
                     fontsize=labelsize)
    ax[2].legend(loc=1, fontsize=legsize)
    plt.show()
    print()
    print()

    # ############# PART D (ANALYTIC SNR) #############
    print('(d) ANALYTIC SNR')
    # get snr for each detector
    esnr_H1 = ut.expect_snr(templ_H1, powers_H1, window=window)
    esnr_L1 = ut.expect_snr(templ_L1, powers_L1, window=window)
    esnr_tot = np.sqrt(esnr_H1**2 + esnr_L1**2)

    # show SNR
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 9))
    ax[0].plot(time-toff, esnr_H1, linewidth=0.5, color='b',
               label='H1 SNR')
    ax[0].set_ylabel(r'SNR', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('Analytic Expected SNR in Time Domain', fontsize=titlesize)
    ax[1].plot(time-toff, esnr_L1, linewidth=0.5, color='r',
               label='L1 SNR')
    ax[1].set_ylabel(r'SNR', fontsize=labelsize)
    ax[1].legend(loc=1, fontsize=legsize)
    ax[2].plot(time-toff, esnr_tot, linewidth=0.5, color='g',
               label='L1 SNR')
    ax[2].set_ylabel(r'SNR', fontsize=labelsize)
    ax[2].set_xlabel(r'GPS Time-{} $\times 10^{{9}}$ s'.format(toff/1e9),
                     fontsize=labelsize)
    ax[2].legend(loc=1, fontsize=legsize)
    plt.show()
    print()
    print()

    # ############# PART E (HALF WEIGHT FREQUENCY) #############
    print('(e) HALF POWER FREQUENCY')
    # get half power frequency for each detector
    hf_H1 = ut.get_hf(freqs, templ_H1, powers_H1, window=window)
    hf_L1 = ut.get_hf(freqs, templ_L1, powers_L1, window=window)
    print("Half frequency for H1: {} Hz".format(hf_H1))
    print("Half frequency for L1: {} Hz".format(hf_L1))
    print()
    print()

    # ############# PART F (TIME OF ARRIVAL) #############
    print('(f) TIME OF ARRIVAL')
    # find snr peak time
    imax_H1 = np.argmax(snr_H1)
    imax_L1 = np.argmax(snr_L1)
    nside = 10

    # get time and uncertainty for each detector
    sguess = 0.001
    ta_H1, eta_H1 = ut.toa(time, snr_H1, sguess=sguess, nside=nside)
    ta_L1, eta_L1 = ut.toa(time, snr_L1, sguess=sguess, nside=nside)
    print('H1 time of arrival: {} ± {}'.format(ta_H1, eta_H1))
    print('L1 time of arrival: {} ± {}'.format(ta_L1, eta_L1))

    # get Gaussian profiles
    prof_H1 = ut.gauss(time[imax_H1-nside:imax_H1+nside], np.max(snr_H1),
                       ta_H1, eta_H1)
    prof_L1 = ut.gauss(time[imax_L1-nside:imax_L1+nside], np.max(snr_L1),
                       ta_L1, eta_L1)

    # Positional uncertainty in the sky => angle.
    # We want to use the difference in time, the speed of light,
    # and the difference in distance to infer an uncertainty in position angle.
    # We use ~3000 km for distance
    # (https://www.ligo.caltech.edu/page/ligo-detectors).
    tdiff = np.abs(ta_H1 - ta_L1) * u.s
    dist = 3e3 * u.km
    c = 3e8 * u.m / u.s
    epos = tdiff * c / dist
    msg = (
        'Typical positional uncertainty:'
        ' {}'.format(epos.to('rad', equivalencies=u.dimensionless_angles()))
        )
    print(msg)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    ax[0].plot(time[imax_H1-nside:imax_H1+nside]-toff,  # SNR near arrival
               snr_H1[imax_H1-nside:imax_H1+nside], 'bo', label='SNR H1')
    ax[0].plot(time[imax_H1-nside:imax_H1+nside]-toff,  # Gaussian profile
               prof_H1, 'g-', label='Gaussian profile (H1)')
    ax[0].set_ylabel('SNR', fontsize=labelsize)
    ax[0].set_title('Gaussian Profiles on SNR Peaks', fontsize=titlesize)
    ax[0].legend(fontsize=legsize)
    ax[1].plot(time[imax_L1-nside:imax_L1+nside]-toff,  # SNR near arrival
               snr_L1[imax_L1-nside:imax_L1+nside], 'ro', label='SNR L1')
    ax[1].plot(time[imax_L1-nside:imax_L1+nside]-toff,  # Gaussian profile
               prof_L1, 'c-', label='Gaussian profile (L1)')
    ax[1].set_ylabel('SNR', fontsize=labelsize)
    ax[1].set_xlabel(r'GPS Time-{} $\times 10^{{9}}$ s'.format(toff/1e9),
                     fontsize=labelsize)
    ax[1].legend(fontsize=legsize)
    plt.show()
