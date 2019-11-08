
import numpy
from matplotlib import pyplot as plt
n=300
rho=numpy.zeros(n)
rho[n/3:(2*n/3)]=1
v=1.0
dx=1.0
x=numpy.arange(n)*dx

plt.ion()
plt.clf()
plt.axis([0,n,-5,5.1])
plt.plot(x,rho)
plt.draw()
plt.savefig('advect_initial_conditions.png')

#advect_finite_volume_timestep_coarse.py
dt=1.0
big_rho=numpy.zeros(n+1)
big_rho[1:]=rho
del rho  #we can delete the to save space
oversamp=0.5 #let's do coarser timestamps
dt_use=dt/oversamp
for step in range(0,15):

    big_rho[0]=0
    drho=big_rho[1:]-big_rho[0:-1]
    big_rho[1:]=big_rho[1:]-v*dt_use/dx*drho

    plt.clf()
    plt.axis([0,n,-5,5.1])
    #plt.axis([0,n,0,1.1])
    plt.plot(x,big_rho[1:])
    plt.draw()
    plt.pause(0.01)
