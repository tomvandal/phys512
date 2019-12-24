"""
Fit a Lorentzian to noisy data using matched filter and Newton's method.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

import utils as ut

# matplotlib settings (looks nicer)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', figsize=(10, 7))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)


def lorentz(x, p):
    """Lorentz function
    Args:
        x  (array): values where we evaluate the function
        p (array): parameters
          - a  (float): amplitude
          - x0 (float): center
          - w  (float): width
    Returns:
        lo (array): simplified Lorentz functino at x
    """

    return p[0] / (((x-p[1])/p[2])**2 + 1)


def grad_lorentz(x, p):
    g = np.zeros([len(x), len(p)])

    g[:, 0] = 1 / (((x-p[1])/p[2])**2 + 1)
    g[:, 1] = 2*p[0]*p[2]**2*(x-p[1]) / (x**2 - 2*x*p[1] + p[2]**2+p[1]**2)**2
    g[:, 2] = 2*p[0]*p[2]*(x-p[1])**2 / (p[1]**2 - 2*x*p[1] + p[2]**2+x**2)**2

    return g


# dataset
y = np.loadtxt('./lorentz_data.txt')
x = np.arange(y.size)
print('There are {} data poins with mean {:.4f} and RMS {:.4f}'.format(
                                                                y.size,
                                                                np.mean(y),
                                                                np.std(y)))
aguess = 0.7
x0 = np.mean(x)
wi = 100  # random, we'll loop anyway
plt.plot(x, y, label='Data signal')
plt.plot(x, lorentz(x, [aguess, x0, wi]), label='Lorentzian template example')
plt.title('Data and Template example', fontsize=20)
plt.legend(fontsize=16)
plt.show()

# ########### part (A) ###########
# get white noise model: smoothed power spectrum
window = windows.cosine(y.size)
# window = np.blackman(y.size)   # I also tested this, worked OK too
powers = ut.powerspec(y, window=window, smooth_sig=1)

# loop through templates with different widths
wvals = np.arange(10, 2000+1)
apeak = np.zeros(wvals.size)
x0peak = np.zeros(wvals.size)
for i in range(wvals.size):
    # generate template
    templ = lorentz(x, [aguess, x0, wvals[i]])

    # matched filter
    mf = ut.matchedfilt(y, templ, powers, window=window)

    # append stuff
    apeak[i] = np.max(mf)
    x0peak[i] = x[np.argmax(mf)]  # x are indices, but in case they change...

# save exploration values
np.savetxt('./amplitudes.txt', apeak)
np.savetxt('./centers.txt', x0peak)

# best guesses
imax = np.argmax(apeak)
wguess = wvals[imax]
x0guess = x0peak[imax]

# plot exploration result
fig, axs = plt.subplots(1, 2)

axs[0].plot(wvals, apeak, label='Amplitude peaks')
axs[0].axvline(wguess, linestyle='--', color='r')
axs[0].set_xlabel('Width', fontsize=18)
axs[0].set_ylabel('Filter Amplitude Peak', fontsize=18)

axs[1].plot(wvals, x0peak, label='Peak centers')
axs[1].axvline(wguess, linestyle='--', color='r')
axs[1].set_xlabel('Width', fontsize=18)
axs[1].set_ylabel('Peak center', fontsize=18)
fig.suptitle('Width Exploration with Matched filters')
plt.show()


# ########### part (B) ###########
# guess from matched filter
print('Strating Guess: a={}, x0={}, w={}'.format(aguess, x0guess, wguess))

pguess = np.array([aguess, x0guess, wguess])

p, pcov = ut.newton(lorentz, grad_lorentz, x, y, pguess, yerr=None, maxit=10,
                    cstol=1e-3, dptol=1e-3)
np.set_printoptions(precision=4)
print('Best-fit parameters:', p)
print('Uncertainties', np.sqrt(np.diag(pcov)))

# plot final result
plt.plot(x, y, label='Data signal')
plt.plot(x, lorentz(x, pguess), 'k--', label='Guess from Filter')
plt.plot(x, lorentz(x, p), 'r-', label='Best fit from Newton')
plt.title('Final fit', fontsize=20)
plt.legend(fontsize=16)
plt.show()
