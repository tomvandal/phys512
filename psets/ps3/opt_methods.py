"""
Useful methods to find best model parameters, including optimizers, samplers,
and differentiation.
"""
import warnings
import time

import numpy as np


def grad_diff1(fun, p, pred, dp=1e-2):
    """One sided first order differentiator

    Use one-sided first order difference to get gradient with respect to
    parameters (made to be used with newton_lm).

    fun:  function to differentiate
    p:    parameter values
    pred: function evaluated at p
    dp:   step to take in parameter values
          (can be scalar or array of length of p)
    """
    if np.isscalar(dp):
        dp = dp * np.ones(len(p))
    else:
        assert len(dp) == len(p), "diff array should have same len as p"
    g = np.zeros([len(pred), len(p)])
    for i in range(len(p)):
        dpi = np.zeros(len(p))
        dpi[i] = dp[i]
        g[:, i] = (fun(p+dpi) - pred) / dp[i]

    return g


def lma_opt(fun, grad_diff, x, y, yerr, pguess, maxit=10, tol=1e-3,
            lamb_init=0.001, dp_grad=None):
    """Levenberg-Marquardt optimizer

    Use LM algorithm to optmize Model on a given dataset.

    Args:
        fun:          function to fit with positional arguments (x, parameters)
        grad_diff:    function to compute gradient by finite differences
        x:            x data to fit
        y:            y data to fit
        yerr:         error on y data
        pguess:       initial guess on parameters (should be nonzero!)
        maxit:        maximum number of iterations
        tol:          maximum chi-sq change for convergence
        lamb_init:    initial value of lambda for LM
        dp_grad:      Steps to take for each parameter. 1/1000 of each
                      parameter if None (default).

    Returns (arrays):
        pars: optimized parameters
        cov:  estimate of the parameters covariance
    """
    # initialize variables
    pguess = np.asarray(pguess)  # ensure numpy array
    pguess = np.where(pguess != 0, pguess, 1e-12)  # ensure all nonzero
    if dp_grad is None:  # if none, use fraction, else handled by function
        dp_grad = pguess * 1e-3
    pars = pguess.copy()
    invnoise = np.matrix(np.diag(1/yerr**2))  # assume uncorr Gauss. noise

    # prediction for guess
    pred = fun(pars)
    res = np.matrix(y-pred).T
    grad = np.matrix(grad_diff(fun, pars, pred, dp=dp_grad))
    chisq = (res.T * invnoise * res).item()
    print("Initial parameters:", pars)
    print("Initial Chi2:", chisq)
    print()

    # newton steps
    lamb = lamb_init
    print("Starting optimization...")
    tstart = time.time()
    for j in range(maxit):

        # solve linear system
        lhs = grad.T * invnoise * grad
        lhs += lamb * np.diag(np.diag(lhs))  # add LM step to diagonal
        rhs = grad.T * invnoise * res
        dpars = np.linalg.inv(lhs) * rhs
        dpars = np.asarray(dpars).flatten()

        # predictive fit with new parameters
        pred_new = fun(pars+dpars)
        res_new = np.matrix(y-pred_new).T
        grad_new = np.matrix(grad_diff(fun, pars+dpars, pred_new, dp=dp_grad))
        chisq_new = (res_new.T * invnoise * res_new).item()
        print("Params for step {}:".format(j+1), pars+dpars)
        print("Chi2:", chisq_new)

        # convergence check
        # if converged, parameters did not significantly change, so no need to
        # assign to new param values
        dchisq = np.abs(chisq_new - chisq)
        if dchisq < tol:
            print("The Newton Method converged after {} iterations".format(j))
            break
        if j == maxit-1:
            msg = ("The maximum of {} iterations was reached before"
                   "convergence... parameters may be poorly constrained."
                   ).format(maxit)
            warnings.warn(msg, RuntimeWarning)
            break

        # LM update lambda
        if chisq_new >= chisq:  # if bad keep increasing lambda
            print("These parameters were REJECTED. Increasing labmda by 10.")
            lamb *= 10
        else:                   # if good, lower lambda and update values
            lamb *= 0.1
            pars += dpars
            pred = pred_new
            res = res_new
            grad = grad_new
            chisq = chisq_new
        print()

    # after opt, find covariance of parameters
    invcov = grad.T * invnoise * grad
    cov = np.linalg.inv(invcov)
    print("The optimization took {} minutes\n".format((time.time()-tstart)/60))

    return np.asarray(pars), np.asarray(cov)


def get_cov(err, p, fun, grad_diff, dp_grad=None):
    """Calculate covariance matrix

    Calculate covariance matrix of a given function's parameters for data with
    gaussian uncorrelated error.

    Args:
        err:       Gaussian uncorrelated error
        p:         parameters
        fun:       function
        grad_diff: method to calculate derivative (Args fun, p, and dp_grad)
        dp_grad:   dp_grad
    Returns covariance matrix
    """
    pred = fun(p)
    if dp_grad is None:
        dp = p * 1e-3
    grad = np.matrix(grad_diff(fun, p, pred, dp=dp))
    invnoise = np.matrix(np.diag(1/err**2))
    cov = np.linalg.inv(grad.T*invnoise*grad)
    cov = np.asarray(cov)

    return np.asarray(cov)


def mcmc(lnprob, p0, proposal, nburn=None, nsteps=1000, savepath=None,
         progsave=False):
    """MCMC MH sampler

    MH algorithm sampling a log-prob with a given proposal distribution
    Note: lnprob = -0.5*chi2

    Args:
        lnprob:   log of the posterior probability, takes parameter vector as
                  only argument
        proposal: proposal distribution giving new parameters, takes parameter
                  vector as only argument
        p0:       initial parameter guess
        nburn:    number of burn in steps (default is 0)
        nsteps:   number of steps in the final chain
        savepath: path of file where chains are saved
        progsave: save after each step if true

    Returns:
        chain: convergence chain of the walker in parameter space
    """
    # initialize
    p0 = np.asarray(p0)
    p = p0.copy()
    lnp = lnprob(p)
    if nburn is None:
        nburn = 0

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


def draw_cov(p, covmat=None, scale_factor=1.0):
    """Draw samples around p with a given covariance matrix
    Args:
        p:
        covmat: covariance matrix. (default is identity)
        scale_factor: general scale factor applied to step on all params.
                      (Default is 1.0)
    """
    if covmat is not None:
        covmat = np.asarray(covmat)
        assert len(np.diag(covmat)) == len(p), ("Cov. mat should be"
                                                "len(p) by len(p) array")
    else:
        covmat = np.eye(len(p))
    achol = np.linalg.cholesky(covmat)

    return p + scale_factor * np.dot(achol, np.random.randn(achol.shape[0]))


def getflat(chains):
        """
        Get flatchain from MCMC chain array (for corner plot, e.g.)
        """
        s = chains.shape
        return np.copy(chains.reshape(s[0] * s[1], s[2]))
