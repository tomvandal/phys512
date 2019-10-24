"""
Sample parameter space using MCMC
"""
import numpy as np

from cmb_methods import get_spectrum
import opt_methods as om
import plots as pl
import varia.misc as vm  # REMOVE WHEN SUBMITTING


def loglike(p):
    # Log likelihood for get_spectrum at give p vector, with WMAP data
    fit = get_spectrum(p)
    return -0.5 * np.sum(((power-fit)/epower)**2)


def loggauss(p, mu, sigma):
    lg = -0.5 * ((p - mu) / sigma)**2 - 0.5*np.log((sigma**2)*2.*np.pi)

    return lg


def logpost_gausstau(p):
    # log posterior prob for loglike with Gaussian prior on tau
    lp = loglike(p)  # without prior, just loglike

    # tau still has to be positive
    if p[3] < 0:
        return - np.inf  # i.e. prob is 0

    # gaussian prior
    lp += loggauss(p[3], 0.0544, 0.0073)

    return lp


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
print()

# set param guess and cov matrix for proposal distribution
pguess = np.array([65.0, 0.02, 0.1, 0.05, 2e-9, 0.96])
# pguess = popt.copy()
print("Initial parameters:", pguess)
print()

# mcmc tunable params
savedir = "mcmc4_guess_newscale_10k/"
savedir = vm.create_dir(savedir)
scale = 0.25  # proposal distribution scale factor
nburn = 0
nsteps = 10000

# run mcmc
chains = om.mcmc(logpost_gausstau, pguess,
                 lambda p: om.draw_cov(p, covmat=pcov, scale_factor=scale),
                 nburn=nburn, nsteps=nsteps, savepath=savedir+"chains.txt",
                 progsave=True)

# plots showing chains
plotchains = chains.copy()
plotchains[:, 4] *= 1e9  # make As readable on plots
plabels = [r"$H_0$", r"$\omega_b h^2$", r"$\omega_c h^2$", r"$\tau$",
           r"$A_s \times 10^9$", r"$n_s$"]
pl.mcmc_chains(np.array([plotchains]), plabels, savepath=savedir+"chains.png",
               show=False)
pl.mcmc_corner(plotchains, plabels, truths=popt, ticksize=12,
               label_kwargs={"fontsize": 20}, savepath=savedir+"corner.png",
               show=False)

# check convergence with FT
ft = np.abs(np.fft.fft(chains, axis=0))
pl.mcmc_chains(np.array([ft]), plabels, show=False, savepath=savedir+"ft.png",
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
            fit_label="MCMC Best Fit", savepath=savedir+"fit.png", show=False)
