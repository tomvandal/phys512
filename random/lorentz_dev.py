import numpy as np
from matplotlib import pyplot as plt

alpha=2.0

n=1000000
myrand=np.random.rand(n)*2*np.pi
mys=np.tan(myrand)

a,b=np.histogram(mys,100,[-20,20])
bb=0.5*(b[1:]+b[:-1])
width=b[1:]-b[:-1]
pred=n*1.0/(1.0+bb**2)*width/np.pi

plt.clf();plt.plot(bb,a)
plt.plot(bb,pred)


