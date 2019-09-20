import numpy as np
from matplotlib import pyplot as plt

n=500
j=np.arange(n)
N=np.zeros([500,500])
for i in range(0,n):
    myvar=(i+j-np.abs(j-i))
    N[:,i]=0.5*myvar

print(np.mean(np.abs(N-N.transpose())))
N=N+1
r=np.linalg.cholesky(N)
Ninv=np.linalg.inv(N)


vec=np.random.randn(n)
noisevec=np.dot(r,vec)

plt.ion()
plt.clf()
plt.plot(noisevec)


sig=5.0
amp=5.0
signal=np.exp(-0.5*( j-0.5*n)**2/sig**2)

dat=amp*signal+noisevec

plt.plot(dat)
lhs=np.dot(signal,np.dot(Ninv,signal))
rhs=np.dot(signal,np.dot(Ninv,dat))
#fitp=np.dot(np.linalg.inv(lhs),rhs)
fitp=rhs/lhs
print('my fit amplitude is ' + repr(fitp))
