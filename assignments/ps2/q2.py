"""
Question 2: Model flare of M-dwarf star observed by HESS telescope.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

# load data
time, flux, _ = np.loadtxt(
    "./229614158_PDCSAP_SC6.txt",
    delimiter=','
                        ).T
# visualize light curve
# plt.figure()
# plt.plot(time, flux, 'k.')
# plt.xlabel("Time (d)")
# plt.ylabel("Flux")
# plt.title("Flaring M-dwark light curve")
# plt.show()

# part (a)
# This model is not linear: we are fitting an exponential and we do not
# approximate to fit some ceofficients like we did in problem 1.


# define model
def expdec(x, p):
    """General Exponential Decay Model

    To avoid degeneracy between the amplitude and the x offset, we combine
    them into one x0 term

    Args:
        x: values where the model is evaluated
        p: array of parameters [x0, b, yoff]
            x0:   term combining the x offset and the amplitude
            b:    decay rate
            yoff: y offset on model

    Returns e^(b(x-x0)) + yoff
    """

    return np.exp(-p[1]*(x-p[0])) + p[2]


# find a region that will work well for modelling
istart = np.argmax(flux)  # where flux is max
iend = np.max(np.argwhere(time <= 1707))  # determined by eye
time = time[istart:iend]
flux = flux[istart:iend]

fluxoff = 1.0  # guessed by eye (when no changes happen flux=1.0)
amp = flux[0] - fluxoff  # flare amplitude
b = 50                   # trial and error guess for the decay rate
t0 = time[0] + np.log(amp) / b  # combine amplitude in exponential
pguess = [t0, b, fluxoff]
print("Initial Guess on parameters:")
print("  Offset and amplitude: t0 = {} ± {}".format(pguess[0]))
print("  Decay rate (1/day):   b = {} ± {}".format(pguess[1]))
print("  Flux offset:          off = {} ± {}".format(pguess[2]))
eguess = expdec(time, pguess)

plt.figure()
plt.axvline(x=time[0], linestyle='--',
            label="Starting Point t={:.2f}".format(time[0]))
plt.plot(time, flux, 'k.', label="HESS Data")
plt.plot(time, eguess, 'r', label="Model Guess")
plt.xlabel("Time (d)")
plt.ylabel("Flux")
plt.title("Isolated Flare Region and Initial Model Guess")
plt.legend()
plt.show()


# part (b)
def expdec_grad(x, p):
    """Compute Gradient of expdec_wrap function
    Args:
        x: values where the model is evaluated
        p: array of parameters [x0, b, yoff]
            x0:   term combining the x offset and the amplitude
            b:    decay rate
            yoff: y offset on model
    Returns:
        g: gradient of expdec at each value x with shape (len(x), len(p))
    """
    g = np.zeros([len(x), len(p)])
    # g[:, 0] = np.exp(-p[2]*(x-p[1]))
    # g[:, 1] = p[0] * p[2] * np.exp(-p[2]*(x-p[1]))
    # g[:, 2] = -(x-p[1]) * p[0] * np.exp(-p[2]*(x-p[1]))
    # g[:, 3] = 1.0

    g[:, 0] = p[1] * np.exp(-p[1]*(x-p[0]))
    g[:, 1] = -(x-p[0]) * np.exp(-p[1]*(x-p[0]))
    g[:, 2] = 1.0

    return g


def newton(fun, gradfun, x, y, pguess, maxit=10, cstol=1e-3, dptol=1e-3):
    """Use newton methods to fit a function
    Args:
        fun:     function to fit with positional arguments (x, parameters)
        gradfun: gradient of fun with positional argumetns (x, parameters)
        x:       x data to fit
        y:       y data to fit
        pguess:  initial guess on parameters
        maxit:   maximum number of iterations
        cstol:   maximum chi-sq relative change for convergence
        dpar:    maximum parameter variation for convergence

    Returns:
        pars: optimized parameters
        cov:  estimate of the parameters covariance
    """
    pguess = np.array(pguess)  # make sure numpy array
    pars = pguess.copy()

    chisq_prev = 1e4  # high dummy chi-sq
    for j in range(maxit):

        pred = fun(x, pars)
        grad = gradfun(x, pars)

        res = y - pred
        chisq = np.sum(res**2)  # no ebars => no weights

        # generate matrix objects
        res = np.matrix(res).T
        grad = np.matrix(grad)

        # solve linear system
        lhs = grad.T * grad
        rhs = grad.T * res
        dpars = np.linalg.inv(lhs) * (rhs)
        dpars = np.asarray(dpars).flatten()
        for k in range(pguess.size):
            pars[k] = pars[k] + dpars[k]

        # convergence check
        csdiff = (chisq_prev - chisq) / chisq
        dprel = np.max(np.abs(dpars/pars))
        if j > 0 and csdiff < cstol and dprel < dptol:
            print("The Newton Method converged after {} iterations".format(j))
            break
        if j == maxit-1:
            msg = ("maxiter was reached without convergence... "
                   "params may be poorly constained")
            warnings.warn(msg, RuntimeWarning)

        chisq_prev = chisq

    # estimate of parameters covariance
    finpred = fun(x, pars)
    fingrad = gradfun(x, pars)
    finres = y - finpred
    invnoise = np.diag(finres**(-2))   # assume noise on each pt is residuals
    tmp = np.dot(fingrad.T, invnoise)  # compute covariance matrix
    invcov = np.dot(tmp, fingrad)
    cov = np.linalg.inv(invcov)

    return pars, cov


p, cov = newton(expdec, expdec_grad, time, flux, pguess)
perr = np.sqrt(np.diag(cov))
print("The best fit parameters are:")
print("  Offset and amplitude: t0 = {} ± {}".format(p[0], perr[0]))
print("  Decay rate (1/day):   b = {} ± {}".format(p[1], perr[1]))
print("  Flux offset:          off = {} ± {}".format(p[2], perr[2]))
# looks reasonable

fit = expdec(time, p)

plt.figure()
plt.axvline(x=time[0], linestyle='--',
            label="Starting Point t={:.2f}".format(time[0]))
plt.plot(time, flux, 'k.', label="HESS Data")
plt.plot(time, eguess, 'r', label="Model Guess")
plt.plot(time, fit, 'b', label="Newton Best Fit")
plt.xlabel("Time (d)")
plt.ylabel("Flux")
plt.title("Isolated Flare Region")
plt.legend()
plt.show()

# I don't trust the errors: there seems to be correlated noise that we did not
# account for
