"""
Methods to extract simple information from ligo data
"""
import h5py
import numpy as np


def readfile(fpath):
    """Read HDF5 LIGO data file
    Args:
        fpath: path to the file containing the data
    Returns:
        strain: strain from data
        dt: (even) time steps in data
    """
    datfile = h5py.File(fpath, 'r')
    dqinfo = datfile['quality']['simple']
    qmask = dqinfo['DQmask'][...]

    meta = datfile['meta']
    gps_start = meta['GPSstart'][()]
    gps_end = gps_start + len(qmask)
    duration = meta['Duration'][()]
    strain = datfile['strain']['Strain'][()]
    dt = (1.0 * duration) / len(strain)
    time = np.arange(gps_start, gps_end, dt)

    datfile.close()

    return strain, time


def readtemp(fpath):
    """Read LIGO waveform template
    Args:
        fpath: path to the file containing the data
    Returns:
        th: template for Handford
        tl: tempate for Livingston
    """
    datfile = h5py.File(fpath, 'r')
    template = datfile['template']
    th = template[0]
    tl = template[1]

    return th, tl


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
    strain_H1, time = readfile(datadir+fn_H1)
    strain_L1, _ = readfile(datadir+fn_L1)
    strains = (strain_H1, strain_L1)

    # load template for each detector
    templ_H1, templ_L1 = readtemp(datadir+fn_template)
    templs = (templ_H1, templ_L1)

    return time, fs, strains, templs
