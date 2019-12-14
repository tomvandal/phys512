"""
Interpolation methods
"""
import numpy as np
from scipy import interpolate


def interp(xdat, ydat, xnew, deriv_true=None):
    """Spline interpolation

    A cubic spline (scipy) is used to perform the interpolation. A (rough)
    estimate of the error can be obtained by comparing linear extrapolations of
    the known (true) derivatives and the spline derivative.

    Args:
        xdat       (array): x data.
        ydat       (array): y data.
        xnew       (array): x values where we will interpolate.
        deriv_true (array): True derivative used to calculate a rough error
                            estimation. Ignored if None (default).
    Returns:
        ynew (array): interpolated y values
        err  (array): estimated error on ynew, obtained by comparing the
                      derivative. Returned only when deriv_true is not None.
    """
    # check arguments
    xdat = np.array(xdat)
    ydat = np.array(ydat)
    xnew = np.array(xnew)
    assert ydat.shape == xdat.shape, 'ydat must have same shape as xdat'
    if deriv_true is not None:
        deriv_true = np.array(deriv_true)
        assert deriv_true.shape == xdat.shape, ('ydat must have same shape as'
                                                ' xdat')

    # use scipy to obtain spline
    spln = interpolate.splrep(xdat, ydat)  # default is cubic (k=3, see docs)

    # evaluate the spline (ynew)
    ynew = interpolate.splev(xnew, spln)

    # evaluate error at each point if needed
    if deriv_true is not None:
        # get first derivative of the spline at known x
        yp = interpolate.splev(xdat, spln, der=1)

        # get x, deriv, and yp at closest xknwon for each xnew
        iprox = []
        for k in range(len(xnew)):
            iprox.append(np.abs(xnew[k] - xdat).argmin())
        x_prox = xdat[iprox]     # known x
        deriv_prox = deriv_true[iprox]  # given true deriv
        yp_prox = yp[iprox]        # spline derivative

        # get rough estimate of the error. At given point x with closest known
        # point x0, f0:
        # f' \approx \Delta f /\Delta x
        # => abs(f_s' - f_t') =  abs((f_s - f0 - f_t + f_0)/(x-x0))
        # => abs(f_s - f_t) = abs((f_s' - f_t')/(x-x0))
        err = np.abs((yp_prox-deriv_prox)*(xnew-x_prox))

        return ynew, err

    return ynew
