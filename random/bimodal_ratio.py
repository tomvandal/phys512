import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def my_randn():
    while True:
        u=2*np.random.rand()
        v=(2*np.random.rand()-1)*2

        rat=v/u
        #if u<np.sqrt(np.exp(-0.5*rat**2)):
        p1=np.exp(-0.5*(rat-2)**2)
        p2=np.exp(-0.5*(rat+2)**2)
        if u<(np.sqrt(p1+p2)):
            return v/u
    

n=100000
vals=np.zeros(n)
for i in range(n):
    vals[i]=my_randn()
a,b=np.histogram(vals,100)
bb=0.5*(b[:-1]+b[1:])
plt.clf();
plt.plot(bb,a)
width=np.diff(b)
pred=n*width*np.exp(-0.5*bb**2)/np.sqrt(2*np.pi)
plt.plot(bb,pred,'r')
