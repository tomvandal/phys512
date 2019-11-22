import numpy as np
from matplotlib import pyplot as plt

alpha=2.0

n=1000000
myrand=np.random.rand(n)
mys=myrand**(1.0/(1.0-alpha))

a,b=np.histogram(mys,100)
bb=0.5*(b[1:]+b[:-1])
width=b[1:]-b[:-1]

plt.clf();plt.loglog(bb,a)
