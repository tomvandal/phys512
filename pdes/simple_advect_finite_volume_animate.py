#simple_advect_finite_volume.py
import numpy
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

n=300
#rho=numpy.zeros(n)
#rho[n/3:(2*n/3)]=1
#x=numpy.arange(n)*dx

x=numpy.arange(n);x=x-0.5*n
rho=numpy.exp(-0.5*(x/20)**2)


v=1.0
dx=1.0


plt.ion()
plt.clf()
#plt.axis([0,n,0,1.1])
plt.plot(x,rho)
plt.draw()
plt.savefig('advect_initial_conditions.png')


#im=plt.imshow(grid,animated=True,vmax=1e-4)

plt.clf();
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,300), ylim=(0, 1.2))
line, = ax.plot([],[], '-', lw=2)


def animate_curve(crud):
    global rho
    global x
    dt=1.0
    for step in range(0,50):
        time.sleep(0.5)
        drho=rho[1:]-rho[0:-1]
        rho[1:]=rho[1:]-v*dt/dx*drho
        line.set_data(x,rho)


ani = animation.FuncAnimation(fig, animate_curve, numpy.arange(50),interval=25, blit=False)
#ani = animation.FuncAnimation(fig, animate_curve,interval=2000, blit=False)
plt.show()
plt.savefig('crud.gif')


#dt=1.0
#for step in range(0,50):
#    #take the difference in densities
#    drho=rho[1:]-rho[0:-1]
#    #update density.  We haven't said what happens at
#    #cell 0 (since cell -1 doesn't exist), ignore for now
#    rho[1:]=rho[1:]-v*dt/dx*drho
#    plt.clf()
#    #plt.axis([0,n,0,1.1])
#    plt.plot(x,rho)
#    #plt.draw()

#ani=animation.FuncAnimation(fig, update_fig, interval=50, blit=False)
