"""
Methods to perform integration
"""
import numpy as np


# function defined in class
def lazy_integrate(fun, a, b, tol):
    x = np.linspace(a, b, 5)
    # dx = (b-a)/4.0  # assigned but unused
    # np.median(np.diff(x))
    y = fun(x)
    neval = len(x)  # let's keep track of function evaluations
    f1 = (y[0]+4*y[2]+y[4])/6.0*(b-a)
    f2 = (y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12.0*(b-a)
    myerr = np.abs(f2-f1)
    # print([a, b, f1, f2])
    if (myerr < tol):
        return (16.0*f2-f1)/15.0, myerr, neval
    else:
        mid = 0.5*(b+a)
        f_left, err_left, neval_left = lazy_integrate(fun, a, mid, tol/2.0)
        f_right, err_right, neval_right = lazy_integrate(
                                                        fun, mid, b, tol/2.0)
        neval = neval + neval_left + neval_right
        f = f_left+f_right
        err = err_left+err_right
        return f, err, neval


def _simp2intv(fun, a, fa, b, fb):
    """Simpson's rule with two intervals
    Evaluate Simpson's Rule with the interval [a,b] splitted in
    two intervals (private method).
    Args:
        fun (callable): function to inegrate
        a (float):      lower bound
        fa (float):     function eval at a
        b (float):      upper bound
        fb (float):     function eval at b
        neval (int):    number of fun eval performed
                        before entering this method
    Returns:
        c (float):     midpoint between a and b
        fc (float):    function eval at c
        s (float):     simpson's rune evaluated with n=2 intervals
        neval (int):   number of function evaluations
    """
    c = (a + b) / 2.0
    fc = fun(c)  # one function integral!
    while np.any(np.isnan(fc)):  # eval nbhd of nan
        c += 1e-13
        fc = fun(c)

    s = (fa + 4*fc + fb) * (b-a)/6.0
    return (c, fc, s)


def _integ_simp(fun, a, fa, b, fb, c, fc, tot, tol, maxcalls=1000, calls=0):
    """Recursive method for Simpson's rule.
    Instead of directly using a linspace and calling function eveytime, we
    will store the values and reuse them directly in this method. As we see
    below, only two function evaluations from calling _simp2intv, instead of
    5 per call as above (or even 6 per call for a less efficient method).
    Args:
        fun (callable): function to inegrate
        a (float):      lower bound
        fa (float):     function eval at a
        b (float):      upper bound
        fb (float):     function eval at b
        c (float):      midpoint between a and b
        fc (float):     function eval at c
        tot (float):    evaluation of simpson rule for two sections delimited
                        by a,c,b.
        neval (int):    number of fun eval performed
                        before entering this method
        tol (float):    tolerance on the precision
        maxcalls (int): max number of recursive calls
        calls (int):    number of calls made before entering this one
    Returns:
        integ (float): value of the evaluated integral
        myerr (float): estimated error on the integral evaluation
        neval (int):   number of function evaluations
    """
    # left and right subintervals
    cl, fcl, left = _simp2intv(fun, a, fa, c, fc)
    cr, fcr, right = _simp2intv(fun, c, fc, b, fb)
    neval = 2  # one for each of the above
    split = left+right  # combine left and right
    myerr = np.abs(split - tot)  # equiv to f2-f1 of lazy method
    if np.any(np.isnan(myerr)):  # sanity check for exploration methods
        raise RuntimeError("Error array contains NaNs.")
    if np.all(myerr < tol):  # TOL OR 15*TOL???
        integ = (16*split - tot) / 15.0
        return integ, myerr, neval

    # we add an explicit stopping condition to avoid crashing computer
    if calls < maxcalls:
        integ_left, lerr, leval = _integ_simp(fun, a, fa, c, fc,
                                              cl, fcl, left, tol/2.0,
                                              maxcalls=maxcalls,
                                              calls=calls+1)
        integ_right, rerr, reval = _integ_simp(fun, c, fc, b, fb,
                                               cr, fcr, right, tol/2.0,
                                               maxcalls=maxcalls,
                                               calls=calls+1)
        integ = integ_left + integ_right
        myerr = lerr + rerr
        neval = neval + leval + reval

        return integ, myerr, neval
    else:
        raise RuntimeError('maximum number of recursive calls reached')


def eff_integrate(fun, a, b, tol, maxcalls=1000):
    """Efficient adaptative Simpson's integrator

    Args:
        fun (callable): function to inegrate
        a (float):      lower bound
        b (float):      upper bound
        tol (float):    tolerance on the precision
        maxcalls (int): maximum number of recursive calls
    Returns:
        integ (float): value of the integral between a and b
        myerr (float): estimated error on the integral evaluation
        neval (int):   total number of function evaluations
    """
    fa, fb = fun(a), fun(b)  # store fun values to reuse in recursion
    while np.any(np.isnan(fa)):  # eval nbhd of nan
        b += 1e-13
        fa = fun(a)
    while np.any(np.isnan(fb)):  # eval nbhd of nan
        b -= 1e-13
        fb = fun(b)
    c, fc, tot = _simp2intv(fun, a, fa, b, fb)
    integ, myerr, neval = _integ_simp(fun, a, fa, b, fb, c, fc, tot, tol,
                                      maxcalls=maxcalls)
    neval += 3  # from base case evaluations
    return integ, myerr, neval
