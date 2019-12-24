"""
Test for the optimal delta, question 1b of problem set 1.
"""
import numpy as np
import matplotlib.pyplot as plt


def check_delta(x, a, delta_arr=np.logspace(-15, 1, num=50), eps=1e-16,
                plotfile=None):
    """ Script to verify delta optimization

        Test the equations derived above on exponential function.
        This function is just a script allowing to change
        the test parameters easily.

        Args:
            x: where we evaluate the derivative
            a: coefficient multiplying x in the exponential
            delta_arr: values of delta where the equations are tested
            eps: machine precision
                 (python floats have double precision so default=1e-16)

        Prints a short report and produces a plot of E vs delta.
    """
    # Function depending on the parameters of check_delta
    def diff(func, x, delta):
        """Five point differentiator
        Args:
            func: function to differentiate
            x: value at which we take the derivative
            delta: step size

        returns:
            deriv: first derivative of func at x
        """
        deriv = ((8.0*func(x+delta)
                  - 8.0*func(x-delta)
                  - func(x+2.0*delta)
                  + func(x-2.0*delta)) / (12.0*delta))

        return deriv

    def optdelta(func, funcd5, x, eps=eps):
        """Optimal step size for five point differentiator
        Args:
            func: function to differentiate
            funcd5: fifth derivative of func
            x: value where we take the derivative
            eps: machine precision (default: 1e-16, approx double precision)
        Returns:
            delta: best step size to choose to derive func at x
        """
        delta = (45.0 * func(x) * eps / (4.0*funcd5(x)))**0.2

        return delta

    def exp(x, a=a):
        """Exponential function
        Args:
            x: values where we evaluate the function
            a: coefficient mulitplying x in exp
        Returns:
            exp(a*x)
        """
        return np.exp(a*x)

    def expdiff(x, a=a, n=5):
        """Nth derivative of exp(a*x)
        Args:
            x: values where we evaluate the function
            a: coefficient mulitplying x in exp
            n: order of the derivative to be returned
        Returns:
            a**n * exp(a*n)
        """
        return a**n * np.exp(a*x)

    # evaluate error and delta prediction
    deriv_arr = diff(exp, x, delta_arr)
    deriv_true = expdiff(x, n=1)
    err_arr = np.abs(deriv_arr - deriv_true)
    delta_pred = optdelta(exp, expdiff, x)

    # print some info
    print("Derivative of exp({} * x) at x={}".format(a, x))
    print("  Predicted delta:", delta_pred)
    print("  Delta giving minimal error:", delta_arr[np.argmin(err_arr)])
    print()

    plt.plot(delta_arr, err_arr, 'k.', label=r"Calculated error")
    plt.axvline(delta_pred, linestyle='--', label=r"Predicted $\delta$")
    plt.xlabel(r"$\delta$", fontsize=14)
    plt.ylabel(r"$E$", fontsize=14)
    plt.title(r"Error vs. step size", fontsize=18)
    plt.tick_params(labelsize=12)
    plt.xscale("log")
    plt.yscale("log")
    if plotfile is not None:
        plt.savefig(plotfile)
        plt.close()
    else:
        plt.show()


# test at x=0
check_delta(0.0, 1.0, plotfile="q1_x0_a1.png")
check_delta(0.0, 0.01, plotfile="q1_x0_a01.png")

# test at x=100
check_delta(100.0, 1.0, plotfile="q1_x100_a1.png")
check_delta(100.0, 0.01, plotfile="q1_x100_a01.png")
