"""
Simulating time-evolution of hundreds of thousands of particles randomly
scattered with NBody model. Boundary conditions are passed as an argument to
NBody object. Both periodic and non-periodic BCs are implemented.
"""
import nbody as nb
import utils as ut

# Setting up two objects in orbit around center point.
exp2 = 17
npart = 2**exp2
ngrid = 2**9
pos0 = None
vel0 = 0.0
soft = 1.0
dt = 100.0
fpath = None  # './part3.gif'
show = True
m = 1.0/npart
logfile = './energy3.txt'

# NBody model object with parameters set above.
model = nb.NBody(m=m, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0, bc='periodic', ndim=3)

# import matplotlib.pyplot as plt
# plt.imshow(model.get_pot())
# plt.show()

# Animation (.gif file)
niter = 400
title = (r'2$^{{{}}}$ randomly scattered particles for $dt={}$ ({} frames)'
         ).format(exp2,
                  dt,
                  niter)

# ut.grid_animation2d(model, niter=niter, show=show, savepath=fpath,
#                     figsize=None, intv=50, nsteps=1, title=title, repeat=False,
#                     marker='r*', style='grid', logfile=logfile)

ut.density3d(model, niter=niter, show=show, savepath=fpath,
                    figsize=None, intv=50, nsteps=1, title=title, repeat=False,
                    marker='r*', style='grid', logfile=logfile)
