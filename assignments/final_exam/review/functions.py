"""
Useful analytic function defintions
"""
import numpy as np


def exp(x, a, b, c):
        """Exponential function
        General exponential function of the form a*exp(b*x)+c.
        Args:
            x (array): values where we evaluate the function
            a (float): overall amplitude multiplying exp
            b (float): coefficient mulitplying x in exp
            c (float): constant offset
        Returns:
            ex (array): exponential function evaluated at given x values.
        """
        assert np.isscalar(a), 'a must be scalar'
        assert np.isscalar(b), 'b must be scalar'
        assert np.isscalar(c), 'c must be scalar'

        return a*np.exp(b*x) + c


def expdec(x, p):
    """General Exponential Decay Model

    To avoid degeneracy between the amplitude and the x offset, we combine
    them into one x0 term. This is more stable than functions.exp for
    least-squares fitting.

    Args:
        x (array): values where the model is evaluated
        p (array): array of parameters [x0, b, yoff] (floats)
            - x0:   term combining the x offset and the amplitude
            - b:    decay rate
            - yoff: y offset on model

    Returns e^(b(x-x0)) + yoff
    """

    return np.exp(-p[1]*(x-p[0])) + p[2]


def expdec_grad(x, p):
    """Analytic gradient of exponential decay
    Gives the analytic gradient of functions.expdec
    Args:
        x (array): values where the model is evaluated
        p (array): array of parameters [x0, b, yoff] (floats)
            - x0:   term combining the x offset and the amplitude
            - b:    decay rate
            - yoff: y offset on model
    Returns:
        g (array): gradient of expdec at each value x with
                   shape (len(x), len(p))
    """
    g = np.zeros([len(x), len(p)])

    g[:, 0] = p[1] * np.exp(-p[1]*(x-p[0]))
    g[:, 1] = -(x-p[0]) * np.exp(-p[1]*(x-p[0]))
    g[:, 2] = 1.0

    return g


def lorentz(x, x0, gam):
    """Lorentzian function
    General Loretzian function: 1/pi * 0.5*gam / ((x-x0)**2+(0.5*gam)**2)
    Args:
        x   (array): values where we evaluate the function
        x0  (float): center of the peak
        gam (float): scale parameter gamma
    Returns:
        lo (array): Lorentzian evaluated at x
    """
    assert np.isscalar(x0), 'x0 must be a scalar'
    assert np.isscalar(gam), 'gam must be a scalar'

    lo = 0.5*gam / (np.pi * ((x-x0)**2 + (0.5*gam)**2))

    return lo


def lorentz_simple(x):
    """Simplified Lorentz function
    Same as functions.lorentz, without scaling or centering constants
    Args:
        x (array): values where we evaluate the function
    Returns:
        lo (array): simplified Lorentz functino at x
    """

    return 1 / (x**2 + 1)


def gauss_dist(x, mu, sig):
    """Normalized Gaussian distribution
    Args:
        x   (array): values where we evaluate the function
        mu  (float): center of the peak
        sig (float): std. dev.
    Return:
        gau (array): Gaussian evaluated at x
    """
    assert np.isscalar(mu), 'mu must be a scalar'
    assert np.isscalar(sig), 'sig must be a scalar'

    return np.exp(-0.5 * ((x-mu)/sig)**2) / (sig*np.sqrt(2*np.pi))


def loggauss(p, mu, sig):
    """Log of a Gaussian
    Args:
        p   (array): values where we evaluate the function
        mu  (float): center of the peak
        sig (float): std. dev.
    Return:
        lg (array): Log-Gaussian evaluated at p
    """

    lg = -0.5 * ((p - mu) / sig)**2 - 0.5*np.log((sig**2)*2.*np.pi)

    return lg


def gauss(x, a, mu, s):
    """Gaussian function

    Non-normalized Gaussian

    Args:
        x  (array): x input values
        a  (float): amplitude
        mu (float): mean value of x
        s  (float): std. dev. in x
    Returns:
        g (array): y value of the Gaussian at each x
    """
    g = a * np.exp(-0.5*(x-mu)**2/s**2)

    return g
