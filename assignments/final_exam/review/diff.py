"""
Useful differentiation methods
"""
import numpy as np


def diff5pt(func, x, delta):
        """Five point differentiator
        Args:
            func        (callable): Function to differentiate
            x     (array or float): Value(s) at which we take the derivative
            delta (array or float): Step size for differentiation. If float,
                                    differentiate all x values with same step,
                                    otherwise array must have same length as
                                    x.

        returns:
            deriv (float or array): first derivative of func at x
        """
        # Check args
        if np.isscalar(x):
            assert np.isscalar(delta), 'delta must be scalar when x is scalar'
        else:
            x = np.array(x)
            if not np.isscalar(delta):
                delta = np.array(delta)
                assert delta.shape == x.shape, ('delta must have same shape as'
                                                ' x')

        # Take derivative with five point stencil
        deriv = ((8.0*func(x+delta)
                  - 8.0*func(x-delta)
                  - func(x+2.0*delta)
                  + func(x-2.0*delta)) / (12.0*delta))

        return deriv


def optdelta(func, funcd5, x, eps=1e-16):
    """Optimal step size for five point differentiator
    Args:
        func    (callable): Function to differentiate.
        funcd5  (callable): Analytical fifth derivative of func.
        x (float or array): Values where we take the derivative.
        eps        (float): Machine precision. Default is 1e-16, approximate
                            double precision.
    Returns:
        delta (float or array): Best step size to choose to derive func at x.
    """
    delta = (45.0 * func(x) * eps / (4.0*funcd5(x)))**0.2

    return delta


def expdiff(x, a=1.0, b=1.0, n=5):
        """Nth analytical derivative of an exponential function of the form
        a*exp(b*c).
        Args:
            a (float): overall amplitude multiplying exp
            b (float): coefficient mulitplying x in exp
            n: order of the derivative to be returned
        Returns:
            a * b**n * exp(b*n)
        """
        assert np.isscalar(a), 'a must be scalar'
        assert np.isscalar(b), 'b must be scalar'

        return a * b**n * np.exp(b*x)


def pgrad_diff1(fun, p, pred, dp=1e-2):
    """One sided first order differentiator

    Use one-sided first order difference to get gradient with respect to
    parameters (made to be used with optimize.newton_lm, at first).

    fun      (callable): function to differentiate
    p           (array): parameter values
    pred        (array): function evaluated at p
    dp (array or float): step to take in parameter values. If float, multiplies
                         input parameter values
    """
    if np.isscalar(dp):
        dp = dp * np.ones(len(p))
    else:
        assert len(dp) == len(p), "diff array should have same len as p"
    g = np.zeros([len(pred), len(p)])
    for i in range(len(p)):
        dpi = np.zeros(len(p))
        dpi[i] = dp[i]
        g[:, i] = (fun(p+dpi) - pred) / dp[i]

    return g
