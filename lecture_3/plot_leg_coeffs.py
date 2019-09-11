import numpy as np
from matplotlib import pyplot as plt
import legendre

n=5
c=legendre.integration_coeffs_legendre(n)
plt.clf();
plt.plot(c)
plt.xlabel('Index')
plt.ylabel('Weight')
plt.title('Integration Coefficients for n=' + repr(n))
plt.savefig('leg_coeffs_'+repr(n)+'.png')
