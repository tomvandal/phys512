"""
Solution to question 4: integrating to get the E-field of  a spherical shell
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import integrals as integ

# R value and z array
R = 1.0
zvals = np.linspace(1e-3, 5, num=100)  # z values >, < & == R
zvals = np.append(zvals, 1.0)
zvals = np.unique(zvals)


# setting up the problem
def fu(u, z=zvals, R=1.0):
    """Integrand for the E-field (see griffiths problem 2.7 for derivation)
    Args:
        u: values at which we evaluate f
        z: position on the z axis
        R: radius of the shell
    Returns
        f: function value at u
    """
    f = (z - R*u) / (R**2 + z**2 - 2*R*z*u)**(1.5)

    return f


def Efield(z, R=1.0):
    """Analytical soln for E-field along z axis
    Args:
        z: position on the z axis
        R: radius of the shell
    Returns:
        efield: E-field at z
    """
    try:
        efield = np.zeros(len(z))
    except TypeError:
        efield = np.zeros(1)

    # inside stays 0. assign others
    efield[z > R] = z[z > R]**(-2.0)
    efield[z == R] = np.nan * np.ones(len(z[z == R]))

    return efield


def quadloop(fun, a, b, zvals=zvals, args=()):
    """Integrate with quad at several z values
    """
    flist, elist = [], []
    for z in zvals:
        f, e = quad(lambda u: fu(u, z=z), a, b, args=args)
        flist.append(f)
        elist.append(e)

    return np.array(flist), np.array(elist)


# NOTE 1: I adapted the integrator from 3 so that it is compatible with arrays
#         to pass zvals as an argument to f directly

# NOTE 2: There is a singularity at u=1 when z=R. Scipy quad is not affected by
#         this, but our integrator was. I adjusted the integrator so that it
#         explores the nbhd of the u causing problem and that fixed it for this
#         case. However, I am not sure my implementation would be stable for
#         other similar situations (it might need a more extensive
#         exploration).


# integrate fu to get E-field
# the n*pi factors just result from the normalizations I neglected above
a, b = -1, 1  # bounds on u
field_myint, myerr, neval = integ.eff_integrate(fu, a, b, 1e-7, maxcalls=1000)
field_myint *= 2*np.pi  # rescale to be consistent with Efield
field_quad, quaderr = quadloop(fu, a, b)
field_quad *= 2*np.pi
field_def = Efield(zvals) * 4*np.pi  # rescale to be consistent w/ integration

# plotting with z offset to show the shapes
plt.plot(zvals, field_def, "Analytical E")
plt.plot(zvals+2, field_myint, "My integrator (z+2)")
plt.plot(zvals+4, field_quad, "Scipy Quad (z+4)")
plt.xlabe("$z$ + constant", fontsize=14)
plt.ylabel("$E$", fontsize=14)
plt.title("E-field of shell with $R=1$", fontsize=18)
plt.legend(fontsize=14)
plt.show()
