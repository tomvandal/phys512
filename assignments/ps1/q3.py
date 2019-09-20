"""
Tets the integrators from integrals.py
"""
import numpy as np

import integrals as integ

sig = 0.1  # used in f2 def, but needed later


def fun0(x):
    return np.exp(x)


def fun1(x):
    return 1.0/(1.0+x**2)


def fun2(x, sig=sig):
    return 1.0+np.exp(-0.5*x**2/(sig**2))


# From these evaluations, we see that both integrators give similar errors
# wrt the analytic integral. We see that we need about half the functions calls
# compared to the lazy integration method when we use the efficient
# (a bit less as we see when neval is high).
print("Integrating exp")
a, b = -1, 1
pred = np.exp(b) - np.exp(a)
f_lazy, myerr_lazy, neval_lazy = integ.lazy_integrate(fun0, a, b, 1e-3)
f_eff, myerr_eff, neval_eff = integ.eff_integrate(fun0, a, b, 1e-3)
print("  Lazy method: f, myerr, neval= {}, {}, {}. Err={}".format(f_lazy,
                                                                  myerr_lazy,
                                                                  neval_lazy,
                                                                  f_lazy-pred))
print("  Efficient method: f, myerr, neval= {}, {}, {}. Err={}".format(
                                                                  f_eff,
                                                                  myerr_eff,
                                                                  neval_eff,
                                                                  f_eff-pred))

print("Integrating fun1")
a, b = -1, 1
pred = np.arctan(b) - np.arctan(a)
f_lazy, myerr_lazy, neval_lazy = integ.lazy_integrate(fun1, a, b, 1e-4)
f_eff, myerr_eff, neval_eff = integ.eff_integrate(fun1, a, b, 1e-4)
print("  Lazy method: f, myerr, neval= {}, {}, {}. Err={}".format(f_lazy,
                                                                  myerr_lazy,
                                                                  neval_lazy,
                                                                  f_lazy-pred))
print("  Efficient method: f, myerr, neval= {}, {}, {}. Err={}".format(
                                                                  f_eff,
                                                                  myerr_eff,
                                                                  neval_eff,
                                                                  f_eff-pred))

print("Integrating fun2")
a, b = -1, 1
pred = (b-a) + np.sqrt(2*np.pi)*sig
f_lazy, myerr_lazy, neval_lazy = integ.lazy_integrate(fun2, a, b, 1e-4)
f_eff, myerr_eff, neval_eff = integ.eff_integrate(fun2, a, b, 1e-4)
print("  Lazy method: f, myerr, neval= {}, {}, {}. Err={}".format(f_lazy,
                                                                  myerr_lazy,
                                                                  neval_lazy,
                                                                  f_lazy-pred))
print("  Efficient method: f, myerr, neval= {}, {}, {}. Err={}".format(
                                                                  f_eff,
                                                                  myerr_eff,
                                                                  neval_eff,
                                                                  f_eff-pred))
