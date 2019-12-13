"""
Methods for MCMC sampling
"""
import time

import numpy as np


def mcmc(lnprob, p0, proposal, nburn=0, nsteps=1000, savepath=None,
         progsave=False):
    """MCMC MH sampler

    MH algorithm sampling a log-prob with a given proposal distribution
    Note: lnprob = -0.5*chi2 when no priors are applied.

    Args:
        lnprob    (callable): log of the posterior probability, takes parameter
                              vector as only argument
        p0           (array): initial parameter guess
        proposal  (callable): function defining proposal distribution giving
                              new parameters at each step, takes parameter
                              vector as only argument
        nburn          (int): number of burn in steps (default is 0)
        nsteps         (int): number of steps in the final chain (default 1000)
        savepath       (str): path of file where chains will be saved
        progsave       (int): save after each step if true (deafult false)

    Returns:
        chain (array): convergence chain of the walker in parameter space,
                       shape (nstep, npars)
    """
    # check args
    p0 = np.asarray(p0)
    nburn = int(nburn)
    nsteps = int(nsteps)

    # first log prob
    p = p0.copy()
    lnp = lnprob(p)

    # burn in
    if nburn != 0:
        print("Starting burn-in")
    tstart = time.time()
    acc_num = 0
    for i in range(nburn):

        # draw new sample
        pnew = proposal(p)
        lnpnew = lnprob(pnew)

        # check accpetance and update
        acc = lnpnew > lnp  # to avoid gen random when not necessary
        if not acc:
            acc = lnpnew - lnp > np.log(np.random.rand(1))
        if acc:
            p = pnew
            lnp = lnpnew
            acc_num += 1

    if nburn != 0:
        print("Burn in acceptance fraction:", float(acc_num) / nburn)
        print("Burn in took {} minutes".format((time.time()-tstart)/60))
        print()

    # final chain
    acc_num = 0
    print("Starting production chain")
    chains = np.zeros([nsteps, len(p)])
    if savepath is not None and progsave:
        with open(savepath, 'w') as datfile:
            for i in range(nsteps):

                # draw new sample
                pnew = proposal(p)
                lnpnew = lnprob(pnew)

                # check acceptance and update
                acc = lnpnew > lnp  # to avoid gen random when not necessary
                if not acc:
                    acc = lnpnew - lnp > np.log(np.random.rand(1))
                if acc:
                    p = pnew
                    lnp = lnpnew
                    acc_num += 1

                # update chain
                datfile.write("{:s}\n".format(" ".join(str(val) for val in p)))
                chains[i] = p
    else:
        for i in range(nsteps):

            # draw new sample
            pnew = proposal(p)
            lnpnew = lnprob(pnew)

            # check acceptance and update
            acc = lnpnew > lnp  # to avoid gen random when not necessary
            if not acc:
                acc = lnpnew - lnp > np.log(np.random.rand(1))
            if acc:
                p = pnew
                lnp = lnpnew
                acc_num += 1

            # update chain
            chains[i] = p
        if savepath is not None:
            np.savetxt(savepath, chains)

    print("Acceptance fraction:", float(acc_num) / nsteps)
    print("MCMC took {} minutes".format((time.time()-tstart)/60))
    print()

    return chains


def _draw_cov(p, covmat=None, scale_factor=1.0):
    """Draw samples around p with a given covariance matrix
    Args:
        p (array): mean parameter values
        covmat (array): covariance matrix. (default is identity)
        scale_factor: general scale factor applied to step on all params.
                      (Default is 1.0)
    """
    if covmat is not None:
        covmat = np.asarray(covmat)
        assert covmat.shape == (len(p), len(p)), ("Cov. mat should be len(p)"
                                                  " by len(p) array")
    else:
        covmat = np.eye(len(p))
    achol = np.linalg.cholesky(covmat)

    return p + scale_factor * np.dot(achol, np.random.randn(achol.shape[0]))
