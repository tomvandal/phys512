import numpy as np
import matplotlib.pyplot as plt

# y = np.log2(np.linspace(1, 1000, 1000))
mant, exp = np.frexp(np.linspace(1, 100, 100))

plt.plot(mant, exp)
plt.show()