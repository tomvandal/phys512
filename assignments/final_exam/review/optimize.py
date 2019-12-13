"""
Optimization methods
"""
import time
import warnings

import numpy as np

import utils as ut


def _chebmat(x, order):
    """Compute Chebyshev polynomials at x-values for orders between 0 and ord
    Args:
        x (array): sorted x values where each poly is evaluated
        order(int): order of the highest polynomial to evaluate
    Returns:
        p (array): array with shape (len(x), ord) containing the polynomial
                   evaluations. Each row contains all orders for one x value.
    """
    # sanity checks
    order = int(order)  # convert order to integer if not already
    assert order >= 0, "order cannot be negative."
    x = np.unique(x)

    # re-map x to the interval -1,1
    a, b = x[0], x[len(x)-1]
    x = ut.xmap(x, a, b)

    # evaluate chebyshev
    p = np.zeros([len(x), ord+1])
    p[:, 0] = 1.0
    if order > 0:
        p[:, 1] = x
    if order > 1:
        for i in range(1, ord):
            p[:, i+1] = 2*x*p[:, i] - p[:, i-1]

    return p


def _chebfit(x, y, ord):
    """Compute Chebyshev coefficients

     Gives Chebyshev coefficients best fitting noise-free dataset. The
     coefficents are computed using least-squares method.

    Args:
        x   (array): x data
        y   (array): y data
        order(array): max order coefficent to compute
    Returns:
        c (array): coefficents corresponding to Chebyshev best-fit
    """
    # sanity check
    order = int(ord)  # convert order to integer if not already
    assert order >= 0, "order cannot be negative."
    assert ord+1 < len(x), (  # even when true, might need reduce
                            "number of parameters equal to number of points..."
                            " try using a smaller order"
                           )

    # least-squares computation (with no noise, i.e. identity N matrix)
    amat = _chebmat(x, ord)  # design matrix (basis evaluated at x)
    lhs = np.dot(amat.T, amat)
    rhs = np.dot(amat.T, y)
    c = np.dot(np.linalg.inv(lhs), rhs)

    return c


def _chebeval(xnew, c, tol=1e-6):
    """Evaluate Chebyshev series

    Evaluates chebyshev series up to order len(c)-1 from pre-computed coeffs.
    The series is truncated before first coeffcicent with abs value less than
    the tolerance.

    Args:
        xnew (array): x values where the series is computed
        c    (array): Coefficents used to evaluate series
        tol  (float): Tolerance on the error
    Returns:
        ypred (array): Chebyshev fit at xnew
        nterms  (int): Final number of terms needed for the fit (order+1)
    """
    # truncate coeff array
    c = np.array(c)
    try:
        indtrunc = np.argwhere(np.abs(c) < tol).flatten()[0]
        c = c[:indtrunc]
    except IndexError:
        coeff_warn = ("The coefficients did not satisfy tolerance... chebeval"
                      " will use all available coefficients")
        warnings.warn(coeff_warn, RuntimeWarning)
    nterms = len(c)  # final number of terms needed

    # evaluate cheb fit
    pmat = _chebmat(xnew, len(c)-1)  # new polynomial values
    ypred = np.dot(pmat, c)

    return ypred, nterms


def chebpred(x, y, maxord, tol=1e-6):
    """Chebyshev polynomials fit
    Use Chebyshev polynomials to generate predictive fit on data.
    Args:
        x      (array): x data
        y      (array): ypred data
        maxorder(array): Max order coefficent to compute
        tol    (array): Tolerance on the error
    Returns:
        ypred  (array): Chebyshev fit at xnew
        nterms (array): Final number of terms needed for the fit (order+1)
    """
    coeffs = _chebfit(x, y, maxord)
    ypred, nterms = _chebeval(x, coeffs, tol=tol)

    return ypred, nterms


def newton(fun, gradfun, x, y, pguess, maxit=10, cstol=1e-3, dptol=1e-3):
    """Use Newton's method to fit a function to noise-free data
    This methods takes a callable function as an argument and always returns
    both best fit parameters and their covariance matrix.
    Args:
        fun     (callable): function to fit with positional arguments
                            (x, parameters)
        gradfun (callable): gradient of fun with positional argumetns
                            (x, parameters)
        x          (array): x data to fit
        y          (array): y (noise-free) data to fit
        pguess     (array): initial guess on parameters
        maxit      (float): maximum number of iterations
        cstol      (float): maximum chi-sq relative change for convergence
        dpar       (float): maximum parameter variation for convergence

    Returns:
        pars (array): optimized parameters
        cov  (array): estimate of the parameters covariance
    """
    # check args
    pguess = np.array(pguess)  # make sure numpy array
    pars = pguess.copy()
    maxit = int(maxit)
    cstol = float(cstol)
    dptol = float(dptol)

    # Perform Newton solver
    chisq_prev = 1e4  # chi2 initialization
    for j in range(maxit):

        pred = fun(x, pars)
        grad = gradfun(x, pars)

        res = y - pred
        chisq = np.sum(res**2)  # no ebars => no weights

        # generate matrix objects
        res = np.matrix(res).T
        grad = np.matrix(grad)

        # solve linear system
        lhs = grad.T * grad
        rhs = grad.T * res
        dpars = np.linalg.inv(lhs) * (rhs)
        dpars = np.asarray(dpars).flatten()
        for k in range(pguess.size):
            pars[k] = pars[k] + dpars[k]

        # convergence check
        csdiff = (chisq_prev - chisq) / chisq
        dprel = np.max(np.abs(dpars/pars))
        if j > 0 and csdiff < cstol and dprel < dptol:
            print("The Newton Method converged after {} iterations".format(j))
            break
        if j == maxit-1:
            msg = ("maxiter was reached without convergence... "
                   "params may be poorly constained")
            warnings.warn(msg, RuntimeWarning)

        chisq_prev = chisq

    # estimate of parameters covariance
    finpred = fun(x, pars)
    fingrad = gradfun(x, pars)
    finres = y - finpred
    invnoise = np.diag(finres**(-2))   # assume noise on each pt is residuals
    tmp = np.dot(fingrad.T, invnoise)  # compute covariance matrix
    invcov = np.dot(tmp, fingrad)
    cov = np.linalg.inv(invcov)

    return pars, cov


def lma_opt(fun, grad_diff, x, y, yerr, pguess, maxit=10, tol=1e-3,
            lamb_init=0.001, dp_grad=None):
    """Levenberg-Marquardt algorithm optimizer

    Use LM algorithm (LMA) to optmize Model on a given dataset.

    Args:
        fun           (callable): function to fit with positional x and
                                  parameters as positional arguments
        grad_diff     (callable): function to compute parameter gradient by
                                  finite differences taking arguments fun, pars
                                  pred, dp.
        x                (array): x data to fit
        y                (array): y data to fit
        yerr             (array): error on y data
        pguess           (array): initial guess on parameters (should be
                                  nonzero!)
        maxit              (int): maximum number of iterations
        tol              (float): maximum chi-sq change for convergence
        lamb_init        (float): initial value of lambda for LMA alogrithm
        dp_grad (array or float): Steps to take for each parameter. If float,
                                  applied as scaling on each param. 1/1000 of
                                  each parameter if None (default).

    Returns:
        pars (array): optimized parameters
        cov  (array): estimate of the parameters covariance
    """
    # initialize variables
    pguess = np.asarray(pguess)  # ensure numpy array
    pguess = np.where(pguess != 0, pguess, 1e-12)  # ensure all nonzero
    if dp_grad is None:  # if none, use fraction, else handled by function
        dp_grad = pguess * 1e-3
    else:
        dp_grad = np.array(dp_grad)
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
        err              (array): Gaussian uncorrelated error
        p                (array): parameters
        fun              (array): function
        grad_diff        (array): method to calculate derivative
                                  (Args: fun, p, and dp_grad)
        dp_grad (array or float): Steps to take for each parameter. If float,
                                  applied as scaling on each param. 1/1000 of
                                  each parameter if None (default).
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
