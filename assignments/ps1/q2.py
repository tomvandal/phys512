"""
Interpolation routine for problem 2. Tested on lakeshore diode data.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# where plot will be saved
plotfile = None  # "q2.png"


# the routine asked for in the problem set
def interp(spln, xnew, xknown=None, deriv_true=None):
    """Interpolate using a spline (with error estimate)

    Interpolation is performed with scipy.interpolate.splev.

    The (rough) estimate of the error is obtained by comparing linear
    extrapolations of the derivatives, since the only extra info we have about
    the tabulated values is the derivative, and the spline takes the exact
    value at knots.

    Args:
        spln (tuple): tuple with knots, coefficients, and order
                      (as returned by scipy.interpolate.splrep)
        xnew (array): x-values where we want to interpolate the function
                      represented by spln
        xknown (array): x values used to calculate spln (default: None)
        deriv_true (array): true derivative at known
    Returns:
        ynew (array): y-values of the interpolation
        err (array): estimated error at each y value
                       (returned only if known and deriv_true are given)
    """
    # evaluate the spline (ynew)
    ynew = interpolate.splev(xnew, spln)

    # evaluate error at each point if needed
    if xknown is not None and deriv_true is not None:
        # get first derivative of the spline at known x
        yp = interpolate.splev(xknown, spln, der=1)

        # get x, deriv, and yp at closest xknwon for each xnew
        iprox = []
        for k in range(len(xnew)):
            iprox.append(np.abs(xnew[k] - xknown).argmin())
        x_prox = xknown[iprox]     # known x
        deriv_prox = deriv_true[iprox]  # given true deriv
        yp_prox = yp[iprox]        # spline derivative

        # get rough estimate of the error. At given point x with closest known
        # point x0, f0:
        # f' \approx \delta f /\delta x
        # => abs(f_s' - f_t') =  abs((f_s - f0 - f_t + f_0)/(x-x0))
        # => abs(f_s - f_t) = abs((f_s' - f_t')/(x-x0))
        err = np.abs((yp_prox-deriv_prox)*(xnew-x_prox))

        return ynew, err

    return ynew


# load diode data (K, V, mV/K)
temp, volt, dvdt = np.loadtxt(
                "/home/thomas/OneDrive/phys512/problem_sets/lakeshore.txt").T
dvdt *= 1e-3  # adjust derivative

# use scipy to setup spline that we will use later
spln = interpolate.splrep(temp, volt)  # default is cubic (k=3, see docs)


# test an interpolation
xinter = np.linspace(temp.min(),
                     temp.max(),
                     num=10000)  # many pts to model dense region near 0
xinter = np.append(xinter, temp)
xinter = np.unique(xinter)
yinter, einter = interp(spln, xinter, xknown=temp, deriv_true=dvdt)

# mean estimated error (I get around 1e-5)
print("Mean estimated error: {}".format(np.mean(einter)))

# plot of the data along the interpolation
# Note: the error is not visible unless we zoom in
plt.plot(temp, volt, 'k.', label='Lakeshore data')
plt.plot(xinter, yinter, 'r', label='Spline interpolation')
plt.fill_between(xinter, yinter+3*einter, yinter-3*einter,
                 color='r', alpha=0.3,
                 label=r'Spline 3$\sigma$ interval (not visible)')
plt.xlabel('Temperature $T$ (K)', fontsize=14)
plt.ylabel('Voltage $V$ (V)', fontsize=14)
plt.legend(fontsize=14)
if plotfile is not None:
    plt.savefig(plotfile)
    plt.close()
else:
    plt.show()
