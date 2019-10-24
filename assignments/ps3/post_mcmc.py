"""
Manipulate chains from previously completed MCMC run
"""
import numpy as np

import plots as pl
from cmb_methods import get_spectrum


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

# tunable parameters
chainsdir = "mcmc_low/"  # directory where we find the chain and save plots
addburn = 0

# load chains
chains = np.loadtxt(chainsdir+"chains.txt")
chains = chains[addburn:]  # remove extra burn in
print("The chain has {} steps".format(chains.shape[0]))

# plot of chains
plotchains = chains.copy()
plotchains[:, 4] *= 1e9  # make As readable on plots
plabels = [r"$H_0$", r"$\omega_b h^2$", r"$\omega_c h^2$", r"$\tau$",
           r"$A_s \times 10^9$", r"$n_s$"]
pl.mcmc_chains(np.array([plotchains]), plabels,
               savepath=chainsdir+"chains.png", show=True)
pl.mcmc_corner(plotchains, plabels, truths=popt, ticksize=12,
               label_kwargs={"fontsize": 20}, savepath=chainsdir+"corner.png",
               show=True)

# check convergence with FFT
ft = np.abs(np.fft.fft(chains, axis=0))
pl.mcmc_chains(np.array([ft]), plabels, show=True, savepath=chainsdir+"ft.png",
               scale="log", lim=False)

# final model
pvals = np.median(chains, axis=0)  # get median for each param
perr_hi = np.percentile(chains, 84, axis=0) - pvals
perr_lo = pvals - np.percentile(chains, 16, axis=0)
perr = np.mean([perr_hi, perr_lo])
print("Final parameters:")
for i in range(len(pvals)):
    print("  {} = {} + {} - {}".format(pnames[i],
                                       pvals[i],
                                       perr_hi[i],
                                       perr_lo[i]))
cmb_fit = get_spectrum(pvals)
chi2 = np.sum(((power - cmb_fit)/epower)**2)
print("Final chi2:", chi2)

# plot final model
pl.spectrum(index, power, epower=epower, ind_fit=index, power_fit=cmb_fit,
            fit_label="MCMC Best Fit", savepath=chainsdir+"fit.png")
