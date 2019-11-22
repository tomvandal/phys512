import numpy as np
from matplotlib import pyplot as plt


n=1000
myrand=np.random.rand(n)
myexp=-np.log(myrand)

a,b=np.histogram(myexp,100)
bb=0.5*(b[1:]+b[:-1])
width=b[1:]-b[:-1]
pred=np.exp(-bb)*width*n

plt.clf();
plt.plot(bb,a)
plt.plot(bb,pred)
