"""
Code to integrate with normal polynomials
"""
import numpy as np


# ############## PART C ##############
def get_coeffs(k):

    ak = (1 + (-1)**k) / (1 + k)

    return ak


def get_weights(x, n, norm=False):
    # setup matrices
    x = x.copy()
    A = np.repeat([x], n, axis=0).T
    pows = np.arange(n)
    A = np.power(A, pows)
    a = get_coeffs(pows)
    Ainv = np.linalg.pinv(A)

    w = np.dot(a.T, Ainv)

    # if normalized weights: times total range divide by sum
    if norm:
        return (x.max()-x.min()) * w / np.sum(w)

    return w


def integ(x, y, npow, norm=False, verbose=True):
    # we rescale the weights to the interval, because our previous va
    w = get_weights(x, npow, norm=norm)
    area = np.dot(w, y)
    if verbose:
        print('Integration weights: {}'.format(w))
        print('Sum of weights: {:.4f}'.format(np.sum(w)))
        print('Area under the curve is: {:.4f}'.format(area))

    return area


# Test for flat data
# Create 'data'
npts = 9
xmin, xmax = -1.0, 1.0
x = np.linspace(xmin, xmax, num=npts)
y = np.ones(x.shape, dtype=float)

# use same number of powers as pts for now
totarea = integ(x, y, 9, verbose=True)


# ############## PART D ##############
# Still use same number of powers as pts for now
# 3rd order
npts = 3
xmin, xmax = -1.0, 1.0
dx = (x.max()-x.min()) / (npts-1)
x = np.linspace(xmin, xmax, num=npts)
y = np.ones(x.shape, dtype=float)
print('3rd order (dx factored out):', get_weights(x, npts, norm=True))


# 5th order
npts = 5
xmin, xmax = -1.0, 1.0
dx = (x.max()-x.min()) / (npts-1)
x = np.linspace(xmin, xmax, num=npts)
y = np.ones(x.shape, dtype=float)
print('5th order (dx factored out):', get_weights(x, npts, norm=True))


# ############## PART E ##############
def get_npts(npow):
    """
    Ensure that the number of points is of the form npts = n + (n-1)*j, and
    as close as possible to 30
    """
    j = (30 - npow) / (npow - 1)
    npts = npow + np.round(j) * (npow-1)
    return int(npts)


for n in [3, 5, 7, 9]:
    print('FOR n={}'.format(n))
    xmin, xmax = -1.0, 1.0
    npts = get_npts(n)
    x = np.linspace(xmin, xmax, num=npts)
    y = np.exp(x)
    area = integ(x, y, n, norm=True)
    expec = np.exp(1)-np.exp(-1)
    print('Error and Expected value:', area-expec, expec)
    print()
