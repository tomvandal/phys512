"""
Optimize parameters fitting WMAP data with a Levenberg-Marquardt algorithm
"""
import numpy as np

import plots as pl
import opt_methods as om
from cmb_methods import get_spectrum


def get_spec(p, tau=0.05):
    """ Get power spectrum with fixed tau

    Wrapper for get_spectrum with tau fixed and excluded from parameter array
    """
    p = np.insert(p, 3, tau)
    return get_spectrum(p)


# load WMAP data
# we assume the errror on the power to be Gaussian, uncorrelated
wmap_data = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt")
index, power, epower = wmap_data[:, 0], wmap_data[:, 1], wmap_data[:, 2]

# parameter values guess
pguess = np.array([65.0, 0.02, 0.1, 0.05, 2e-9, 0.96])

# optimization
pguess_short = np.append(pguess[:3], pguess[4:])
tauguess = pguess[3]
p, cov = om.lma_opt(get_spec, om.grad_diff1, index, power, epower,
                    pguess_short, maxit=10, tol=1e-3, lamb_init=0.001,
                    dp_grad=None)
print("Optimized parameters:", p)
print("Errors", np.sqrt(np.diag(cov)))

# final params and cov (with tau)
popt = np.insert(p, 3, tauguess)
pcov = om.get_cov(epower, popt, get_spectrum, om.grad_diff1)
np.savetxt("opt_params.txt", popt)  # save results
np.savetxt("opt_cov.txt", pcov)
pnames = ["Ho", "ombh2", "omch2", "tau", "As", "ns"]
print("Optimized Parameters:")
for i in range(len(popt)):
    print("  {} = {} +/- {}".format(pnames[i],
                                    popt[i],
                                    np.sqrt(np.diag(pcov))[i]))

# plot final result
cmb_fit = get_spectrum(popt)
pl.spectrum(index, power, epower=epower, ind_fit=index, power_fit=cmb_fit,
            fit_label="LMA Best Fit")
