"""
Fit WMAP data for a guess of parameters and print the chi2 value
"""
import numpy as np

from cmb_methods import get_spectrum
import plots as pl

# load WMAP data
# we assume the errror on the power to be Gaussian, uncorrelated
wmap_data = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt")
index, power, epower = wmap_data[:, 0], wmap_data[:, 1], wmap_data[:, 2]

# parameter values guess
pguess = np.array([65.0, 0.02, 0.1, 0.05, 2e-9, 0.96])

# get CAMB model for given parameters
cmb = get_spectrum(pguess)  # , lmax_ret=int(index.max()))

# calculate chi2 for the cmb powers
chi2 = np.sum(((power - cmb)/epower)**2)

# print results
pstr = ("Guess Params.: "
        "H0={}, ombh2={}, omch2={}, tau={}, As={}, ns={}".format(*pguess))
chi2str = "Chi2: {}".format(chi2)
print(pstr)
print(chi2str)  # value is around 1588, as expected

# plot of spectrum and data
pl.spectrum(index, power, epower=epower, ind_fit=index, power_fit=cmb,
            fit_label="Model Guess")
