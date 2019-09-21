import numpy as np
from matplotlib import pyplot as plt

sigsqr=3.0

n=1000
N=np.zeros([n,n])
j=np.arange(n)
for i in range(n):
    myvar=sigsqr*(i+j-np.abs(i-j))
    N[:,i]=0.5*myvar

N=N+sigsqr
plt.ion()

amp=20.0
sig=5.0
model=np.exp(-0.5*(j-0.5*n)**2/sig**2)


r=np.linalg.cholesky(N)
vec=np.random.randn(n)
noise=np.dot(r,vec)
simdat=noise+model*amp

lhs=np.dot(model,np.dot(np.linalg.inv(N),model))
rhs=np.dot(model,np.dot(np.linalg.inv(N),simdat))
amp=rhs/lhs
print('fit amplitude is ' + repr(amp) + ' +/- ' + repr(np.sqrt(1/lhs)))
