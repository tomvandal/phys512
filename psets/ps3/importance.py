"""
Use a prior to importance sample MCMC chains
"""
import numpy as np

from cmb_methods import get_spectrum
import plots as pl


def gauss(p, mu, sigma):
    # simple gaussian distribution

    return (1.0/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((p-mu)/sigma)**2)


# load WMAP data
# we assume the errror on the power to be Gaussian, uncorrelated
wmap_data = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt")
index, power, epower = wmap_data[:, 0], wmap_data[:, 1], wmap_data[:, 2]

# load opt results
popt = np.loadtxt("opt_params.txt")
pcov = np.loadtxt("opt_cov.txt")
popt_err = np.sqrt(np.diag(pcov))

# print opt result
pnames = ["Ho", "ombh2", "omch2", "tau", "As", "ns"]
print("Optimized Parameters:")
for i in range(len(popt)):
    print("  {} = {} +/- {}".format(pnames[i],
                                    popt[i],
                                    popt_err[i]))

# tunable_parameters
chainsdir = "mcmc3/"
addburn = 1000
save = False

# load chains
chains = np.loadtxt(chainsdir+"chains.txt")
chains = chains[addburn:]  # remove extra burn in
nsteps = chains.shape[0]
print("The chain has {} steps".format(nsteps))

# without weights
pvals = np.median(chains, axis=0)  # get median for each param
perr_hi = np.percentile(chains, 84, axis=0) - pvals
perr_lo = pvals - np.percentile(chains, 16, axis=0)
perr = np.mean([perr_hi, perr_lo])

# importance sample with Gaussian weigths
# we can use numpy to directly get parameters and uncertainties
weights = gauss(chains[:, 3], 0.0544, 0.0073)
pvals = np.average(chains, axis=0, weights=weights)
p_err = np.sqrt(np.diag(np.cov(chains, rowvar=False, aweights=weights,
                               ddof=0)))
perr_hi = p_err.copy()
perr_lo = p_err.copy()

# print opt result
pnames = ["Ho", "ombh2", "omch2", "tau", "As", "ns"]
print("LMA Parameters:")
for i in range(len(popt)):
    if i == 4:
        print("  {} = {:.4f} +/- {:.4f}".format(
                                            pnames[i]+" x 10^9",
                                            popt[i]*10**9,
                                            np.sqrt(np.diag(pcov))[i]*10**9))
    else:
        print("  {} = {:.4f} +/- {:.4f}".format(pnames[i],
                                                popt[i],
                                                np.sqrt(np.diag(pcov))[i]))

print("Parameters from importance sampling:")
for i in range(len(pvals)):
    if i == 4:
        print("  {} = {:.4f} + {:.4f} - {:.4f}".format(
                                                    pnames[i]+" x 10^9",
                                                    pvals[i]*10**9,
                                                    perr_hi[i]*10**9,
                                                    perr_lo[i]*10**9))
    else:
        print("  {} = {:.4f} + {:.4f} - {:.4f}".format(pnames[i],
                                                       pvals[i],
                                                       perr_hi[i],
                                                       perr_lo[i]))
cmb_fit = get_spectrum(pvals)
chi2 = np.sum(((power - cmb_fit)/epower)**2)
print("Chi2:", chi2)

# plot final model
pl.spectrum(index, power, epower=epower, ind_fit=index, power_fit=cmb_fit,
            fit_label="MCMC Best Fit",
            savepath=chainsdir+"fit.png" if save else None)
