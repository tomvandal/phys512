"""
Files to extract simple information from ligo data
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
