"""
Question 1: Chebyshev polynomial fit and least squares.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt


def xmap(x, a, b):
    """Re-map x range to [-1, 1] intv to allow chebyshev fit

    Use transformation y=(x - 0.5 * (b+a)) / (0.5 * (b-a)) to map an arbitrary
    range to [-1, 1]. (from numerical recipes)

    Args:
        x:   array of x values

    Returns:
        x mapped to [-1, 1]
    """
    bpa = 0.5 * (b + a)
    bma = 0.5 * (b - a)

    return (x - bpa) / bma


def chebmat(x, ord):
    """Compute Chebyshev polynomials at x-values for orders between 0 and ord
    Args:
        x:   sorted x values where each poly is evaluated
        ord: order of the highest polynomial to evaluate
    Returns:
        p: array of shape (len(x), ord) containing the polynomial evaluations.
           Each row contains all orders for one x value.
    """
    # sanity check
    ord = int(ord)  # convert order to integer if not already
    assert ord >= 0, "ord cannot be negative."

    # re-map x to the interval -1,1
    a, b = x[0], x[len(x)-1]
    x = xmap(x, a, b)

    # evaluate chebyshev
    p = np.zeros([len(x), ord+1])
    p[:, 0] = 1.0
    if ord > 0:
        p[:, 1] = x
    if ord > 1:
        for i in range(1, ord):
            p[:, i+1] = 2*x*p[:, i] - p[:, i-1]

    return p


def chebfit(x, y, ord):
    """Compute Chebyshev coefficients best fitting noise-free dataset

    The coefficents are computed using least-squares method.

    Args:
        x:   x data
        y:   y data
        ord: max order coefficent to compute
    Returns:
        c: coefficents corresponding to Chebyshev best-fit
    """
    # sanity check
    ord = int(ord)  # convert order to integer if not already
    assert ord >= 0, "ord cannot be negative."
    assert ord+1 < len(x), (  # even when true, might need reduce
                            "number of parameters equal to number of points..."
                            " try using a smaller order"
                           )

    # least-squares computation (with no noise, i.e. identity)
    amat = chebmat(x, ord)  # design matrix (basis evaluated at x)
    lhs = np.dot(amat.T, amat)
    rhs = np.dot(amat.T, y)
    c = np.dot(np.linalg.inv(lhs), rhs)

    return c


def chebeval(xnew, c, tol=1e-6):
    """Evaluate Chebyshev series

    Evaluates chebyshev series up to order len(c)-1.
    The series is truncated at before first coeffcicent with abs value less
    than the tolerance.

    Args:
        xnew: x values where the series is computed
        c:    coefficents used to evaluate series
        tol:  tolerance on the error
    Returns (tuple):
        ypred:  Chebyshev fit at xnew
        nterms: final number of terms needed for the fit (order+1)
    """
    # truncate coeff array
    try:
        indtrunc = np.argwhere(np.abs(c) < tol).flatten()[0]
        c = c[:indtrunc]
    except IndexError:
        coeff_warn = ("The coefficients did not satisfy tolerance... "
                      "chebeval will use all available coefficients")
        warnings.warn(coeff_warn, RuntimeWarning)
    nterms = len(c)  # final number of terms needed

    # evaluate cheb fit
    pmat = chebmat(xnew, len(c)-1)  # new polynomial values
    ypred = np.dot(pmat, c)

    return ypred, nterms


def chebpred(x, y, maxord, tol=1e-6):
    """Use Chebyshev polynomials to generate predictive fit on data
    This routine simply wraps up the previous methods in one call.
    Args:
        x:       x data
        y:       y data
        maxord:  max order coefficent to compute
        tol:     tolerance on the error
    Returns (tuple):
        ypred:  Chebyshev fit at xnew
        nterms: final number of terms needed for the fit (order+1)
    """
    coeffs = chebfit(x, y, maxord)
    ypred, nterms = chebeval(x, coeffs, tol=tol)

    return ypred, nterms


# part (a)
npts = 100
ord = 50
tol = 1e-6
# "data" to fit
x = np.linspace(0.5, 1.0, num=npts)    # x values evenly sampled
y = np.log2(x)                        # noiseless y "data"

# cheb best fit
ypred, nterms = chebpred(x, y, ord, tol=tol)
res = ypred - y
print("Initial number of terms: {}".format(ord+1))
print("Number of terms needed: {}".format(nterms))

# regular poly best fit
polypars = np.polyfit(x, y, deg=nterms-1)  # order is less than nterms
ypred2 = np.polyval(polypars, x)
res2 = ypred2 - y

# compare the two fits
print("Chebyshev fit has RMSE = {}, Max Err. = {}.".format(
                                                    np.sqrt(np.mean(res**2)),
                                                    np.max(np.abs(res))
                                                       )
      )
print("Regular poly. fit has RMSE = {}, Max Err. = {}.".format(
                                                    np.sqrt(np.mean(res2**2)),
                                                    np.max(np.abs(res2))
                                                       )
      )

plt.figure()
plt.plot(x, y, 'k.', label="True values")
plt.plot(x, ypred, label="Cheby. fit")
plt.plot(x, ypred2, label="Poly. fit")
plt.title(r"Least squares fits on $\log_{2}\left(x\right)$")
plt.legend()
plt.show()

plt.figure()
plt.axhline(linestyle='--', color='k')
plt.plot(x, res, label="Cheby. res.")
plt.plot(x, res2, label="Poly. res.")
plt.title(r"Residuals of least squares fits on $\log_{2}\left(x\right)$")
plt.legend()
plt.show()


# part (b)
# the compatibility with any x range is taken care of by the
# method xmap defined above

# example where we adjust the interval for cheby only
npts = 1000
ord = 100
tol = 1e-6
# "data" to fit
x = np.linspace(0.5, 100, num=npts)    # x values evenly sampled
y = np.log2(x)                        # noiseless y "data"

# cheb best fit
ypred, nterms = chebpred(x, y, ord, tol=tol)
res = ypred - y
print("Initial number of terms: {}".format(ord+1))
print("Number of terms needed: {}".format(nterms))

# regular poly best fit
polypars = np.polyfit(x, y, deg=nterms-1)  # order is less than nterms
ypred2 = np.polyval(polypars, x)
res2 = ypred2 - y

# compare the two fits
print("Chebyshev fit has RMSE = {}, Max Err. = {}.".format(
                                                    np.sqrt(np.mean(res**2)),
                                                    np.max(np.abs(res))
                                                       )
      )
print("Regular poly. fit has RMSE = {}, Max Err. = {}.".format(
                                                    np.sqrt(np.mean(res2**2)),
                                                    np.max(np.abs(res2))
                                                       )
      )

plt.figure()
plt.plot(x, y, 'k.', label="True values")
plt.plot(x, ypred, label="Cheby. fit")
plt.plot(x, ypred2, label="Poly. fit")
plt.title(r"Least squares fits on $\log_{2}\left(x\right)$")
plt.legend()
plt.show()

plt.figure()
plt.axhline(linestyle='--', color='k')
plt.plot(x, res, label="Cheby. res.")
plt.plot(x, res2, label="Poly. res.")
plt.title(r"Residuals of least squares fits on $\log_{2}\left(x\right)$")
plt.legend()
plt.show()

# example where we adjust the interval for both cheby AND polyfit
npts = 1000
ord = 100
tol = 1e-6
# "data" to fit
x = np.linspace(0.5, 100, num=npts)    # x values evenly sampled
y = np.log2(x)                        # noiseless y "data"

# cheb best fit
ypred, nterms = chebpred(x, y, ord, tol=tol)
res = ypred - y
print("Initial number of terms: {}".format(ord+1))
print("Number of terms needed: {}".format(nterms))

# regular poly best fit
xpoly = xmap(x, x[0], x[len(x)-1])
polypars = np.polyfit(xpoly, y, deg=nterms-1)  # order is less than nterms
ypred2 = np.polyval(polypars, xpoly)
res2 = ypred2 - y

# compare the two fits
print("Chebyshev fit has RMSE = {}, Max Err. = {}.".format(
                                                    np.sqrt(np.mean(res**2)),
                                                    np.max(np.abs(res))
                                                       )
      )
print("Regular poly. fit has RMSE = {}, Max Err. = {}.".format(
                                                    np.sqrt(np.mean(res2**2)),
                                                    np.max(np.abs(res2))
                                                       )
      )

plt.figure()
plt.plot(x, y, 'k.', label="True values")
plt.plot(x, ypred, label="Cheby. fit")
plt.plot(x, ypred2, label="Poly. fit")
plt.title(r"Least squares fits on $\log_{2}\left(x\right)$")
plt.legend()
plt.show()

plt.figure()
plt.axhline(linestyle='--', color='k')
plt.plot(x, res, label="Cheby. res.")
plt.plot(x, res2, label="Poly. res.")
plt.title(r"Residuals of least squares fits on $\log_{2}\left(x\right)$")
plt.legend()
plt.show()
