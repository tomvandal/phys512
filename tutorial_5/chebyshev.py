import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
import matplotlib.pyplot as plt

#1st example

np.random.seed(0)

x = np.linspace(-1, 1, 2000)
y = np.cos(x) + 0.3*np.random.rand(2000)
p = np.polynomial.Chebyshev.fit(x, y, 90)

t = np.linspace(-1, 1, 200)
plt.plot(x, y, 'r.')
plt.plot(t, p(t), 'k-', lw=3)
plt.savefig("random.png")
plt.close()

# 2nd example

x = np.arange(0.5, 1, 1e-6)
y = np.log10(x)

fit = Chebyshev.fit(x, y, 4)

plt.plot(x, y)
plt.plot(x, fit(x), label = "Chebyshev 4th Order Fit", ls ='--')
plt.legend()
plt.savefig("cheby_fit.png")
