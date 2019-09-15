import numpy as np


def integrate(fun, a, b, tol=1e-8):
    neval = 5
    x = np.linspace(a, b, num=neval)
    # dx = (b-a) / 4.0  # might be useful if modif
    y = fun(x)
    dumb_integ = (y[0] + 4*y[2] + y[4]) / 6.0 * (b-a)
    less_dumb_integ = (y[0]+4*y[1]+2*y[2] + 4*y[3]+y[4]) / 12.0 * (b-a)
    myerr = np.abs(dumb_integ - less_dumb_integ)

    if myerr < tol:
        return (16*less_dumb_integ - dumb_integ) / 15.0, neval
    xmid = 0.5*(a+b)
    left_integ, left_neval = integrate(fun, a, xmid, tol=tol/2.0)
    right_integ, right_neval = integrate(fun, xmid, b, tol=tol/2.0)
    integral = left_integ + right_integ
    neval = neval + left_neval + right_neval
    return integral, neval


myval = integrate(np.sin, 0, np.pi)
print(myval-2.0)
