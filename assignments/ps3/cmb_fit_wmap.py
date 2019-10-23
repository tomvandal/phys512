"""
Fit WMAP data using the CAMB package to generate the models
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import camb

import myopt as mo
from gprv import plots


def get_spectrum(p, lmax_acc=2000, lmax_ret=1200):
    """Get power spectrum with CAMB
    Args:
        p: parameters H0, ombh2, omch2, tau, As, ns
        lmax_acc: maximum l index for the accuracy of calculations
        lmax_ret: maximum l index for the returned power spectrum
    Returns:
        tt: TT power spectrum for multipole indices 2 (quadrupole) to lmax_ret
    """
    H0 = p[0]
    ombh2 = p[1]
    omch2 = p[2]
    tau = p[3]
    As = p[4]
    ns = p[5]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=0.06,
                       tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_for_lmax(lmax_acc, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_ret)
    cmb = powers['total']
    tt = cmb[:, 0][2:]  # don't need first to indices

    return tt


def get_spec(p, tau=0.05):
    """ Get power spectrum with fixed tau
    Wrapper for get_spectrum with tau fixed and excluded from parameter array
    """
    p = np.insert(p, 3, tau)
    return get_spectrum(p)


def get_spec_tau(tau, p=np.array([65.0, 0.02, 0.1, 2e-9, 0.96])):
    """ Get power spectrum with only tau varying
    Wrapper for get_spectrum with all but tau fixed and excluded from parameter
    array
    """
    p = np.insert(p, 3, tau)
    return get_spectrum(p)


# options
opt = True  # optimize all params at once of both previous options are flase


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
pstr = ("Initial Parameters: "
        "H0={}, ombh2={}, omch2={}, tau={}, As={}, ns={}".format(*pguess))
chi2str = "Initial Chi-squared: {}".format(chi2)
print(pstr)
print(chi2str)  # value is around 1588, as expected
print()  # some spacing

# optimize all parameters
print("Starting optimization...")
# optimize parameters with tau fixed
if opt:
    pguess_short = np.append(pguess[:3], pguess[4:])
    tauguess = np.array([pguess[3]])
    p, cov = mo.newton_lm(get_spec, mo.grad_diff1, index, power, epower,
                          pguess_short, maxit=10, tol=1e-3, lamb_init=0.001,
                          dp_grad=None)
    print("Optimized parameters:", p)
    print("Errors", np.sqrt(np.diag(cov)))

    p_opt = np.insert(p, 3, tauguess)

    # calculate final covariance matrix (including tau!)
    cmb_pred = get_spectrum(p_opt)
    dp = p_opt * 1e-3
    grad = np.matrix(mo.grad_diff1(get_spectrum, p_opt, cmb_pred, dp=dp))
    invnoise = np.diag(1/epower**2)
    pcov = np.linalg.inv(grad.T*invnoise*grad)
    pcov = np.asarray(pcov)

    # print result
    print("Final Parameters (LM):", p_opt)
    print("Errors (LM Cov)", np.sqrt(np.diag(pcov)))

cmb_opt = get_spectrum(p_opt)

# # plot of spectrum and data
# # plt.errorbar(index, power, yerr=epower, fmt='k.', capsize=0)
# plt.plot(index, power, 'k.', label="WMAP Data")  # easier to view w/o ebars
# plt.plot(index, cmb, 'r-', label="Model Guess")
# plt.plot(index, cmb_opt, '-', label="Optimized Model")
# plt.legend()
# plt.show()


# sample posterior prob with MCMC
# pguess = np.array([65.0, 0.02, 0.1, 0.05, 2e-9, 0.96])
pguess = p_opt


# log probabilities for MCMC
def loglike(p):
    cmb = get_spectrum(p)
    return -0.5 * np.sum(((power - cmb)/epower)**2)


def logpost(p):

    # value without priors is loglike
    lp = loglike(p)

    # positive tau prior
    if p[3] < 0:
        return - np.inf  # ensures that the new likelihood is not accepted

    return lp


# lambda p: mo.draw_normal(p, pscales=pscales, scale_factor=1.0),
# lambda p: mo.draw_cov(p, covmat=cov, scale_factor=1.0)
start = time.time()
chains = mo.mh_sampler(
    logpost, pguess,
    lambda p: mo.draw_cov(p, covmat=pcov, scale_factor=0.1),
    nburn=None, nsteps=100, savepath="chainsprog.dat", progsave=True)
print((time.time()-start)/60, "minutes")
np.savetxt("chains3.dat", chains)
print(chains)
pnames = [r"$H_0$", r"$\omega_b h^2$", r"$\omega_c h^2$", r"$\tau$",
          r"$A_s$", r"$n_s$"]
plots.mcmc_chains(np.array([chains]),
                  pnames, savepath=None,
                  show=True, title=None, tsize=22)
plots.mcmc_corner(chains, pnames, show_titles=True, title=None, tsize=22,
                  truths=None, ticksize=12, label_kwargs={"fontsize": 20},
                  savepath=None, show=True, title_kwargs=None, figsize=None)
