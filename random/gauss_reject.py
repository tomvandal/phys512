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


#plt.clf();plt.plot(bb,a)
#plt.plot(bb,pred)
plt.clf()
plt.plot(bb,1.0/(1.0+bb**2))
plt.plot(bb,np.exp(-0.5*bb**2)/np.sqrt(2))

myrand2=np.random.rand(n)
accept=myrand2<(np.exp(-0.5*mys**2)/np.sqrt(2)/(1.0/(1.0+mys**2)))
newdev=mys[accept]
a,b=np.histogram(newdev,50)
bb=0.5*(b[1:]+b[:-1])
plt.clf();plt.plot(bb,a)


