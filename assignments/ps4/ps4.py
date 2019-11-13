"""
Looking for gravitational waves in LIGO data
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import welch

import read_ligo as rl
import powerspec as ps

# matplotlib options
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# set data dir and event list
datadir = './ligo_data/'  # DIRECTORY CONTAINING THE DATA
eventlist = ['GW150914', 'GW151226', 'GW170104', 'LVT151012']

# load json file
filejson = ''.join([datadir, 'BBH_events_v3.json'])
events = json.load(open(filejson, 'r'))

# iterate over all four events and do all steps for each
for eventname in eventlist:
    print()
    print()
    # event name
    print('EVENT {}'.format(eventname))

    # extract some event info
    event = events[eventname]
    print()
    fn_H1 = event['fn_H1']              # Hanford filename
    fn_L1 = event['fn_L1']              # Livingston filename
    fn_template = event['fn_template']  # template for matched filter later
    fs = event['fs']                    # sampling rate (useful for FT stuff)
    tevent = event['tevent']            # GPS time of event
    fband = event['fband']              # useful for bandpassing

    # load data for each detector (time and dt are same for both detectors)
    strain_H1, time, dt = rl.readfile(datadir+fn_H1)
    strain_L1, _, _ = rl.readfile(datadir+fn_L1)
    fs = 1.0 / dt  # sampling frequency
    print("Length of H1 strain: {} pts".format(len(strain_H1)))
    print("Length of L1 strain: {} pts".format(len(strain_L1)))

    # load template for each detector
    templ_H1, templ_L1 = rl.readtemp(datadir+fn_template)
    print("Length of H1 template: {} pts".format(len(templ_H1)))
    print("Length of L1 template: {} pts".format(len(templ_L1)))
    print()

    # adjust time
    toff = time.min()
    time -= toff

    # # plot H1 and L1
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # ax[0].plot(time-toff, strain_H1*1e19, linewidth=0.5, color='b',
    #            label='H1 Data')
    # ax[0].plot(time-toff, strain_L1*1e19, linewidth=0.5, color='r',
    #            label='L1 Data')
    # ax[0].set_ylabel(r'Strain $\times 10^{19}$')
    # ax[0].legend(loc=1)
    # ax[0].set_title('LIGO Hanford (H1)')
    # ax[1].plot(time-toff, templ_H1*1e19, linewidth=0.5, color='b',
    #            label='H1 Template')
    # ax[1].plot(time-toff, templ_L1*1e19, linewidth=0.5, color='r',
    #            label='L1 Template')
    # ax[1].set_ylabel(r'Strain $\times 10^{19}$')
    # ax[1].set_xlabel(r'GPS Time-{} $\times 10^{{9}}$ s'.format(toff/1e9))
    # ax[1].legend(loc=1)
    # plt.show()
    # plt.savefig(eventname+'.png')
    # plt.close(fig)

    # ################ part (a) ################
    # we will use a window function and take the FFT to get a power spectrum
    # then we will plot the ASD (sqrt of power spectrum)

    # powers, freqs = powerspec(strain_H1, t=time)
    powers, freqs = ps.powerspec(strain_H1, fs=4096, winfun=ps.blackwin, nfft=4096*4, noverlap=4096*2)
    freqs_sci, powers_sci = welch(strain_H1, fs=4096, window='hann', nperseg=4*4096, scaling='spectrum')
    asd = np.sqrt(powers)
    asd_sci = np.sqrt(powers_sci)
    plt.loglog(freqs, asd)
    plt.loglog(freqs_sci, asd_sci)
    plt.xlim(20, 2000)
    plt.ylim(1e-24, 1e-19)
    plt.show()