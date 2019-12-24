"""
Simulating time-evolution of hundreds of thousands of particles randomly
scattered with NBody model. Boundary conditions are passed as an argument to
NBody object. Both periodic and non-periodic BCs are implemented.
"""
import numpy as np

import nbody as nb
import utils as ut

# Setting up two objects in orbit around center point.
npart = 2**17
ngrid = 2**9
pos0 = None
vel0 = 0.0
<<<<<<< HEAD
soft = 10
=======
soft = 1
>>>>>>> parent of 0f23b3a... first energy results
dt = 100.0
fpath = None  # './fig3_nonper.gif'
show = True
<<<<<<< HEAD
m = 1.0/npart  # scale mass with npart to limit total energy
logfile = None  # './energy3_nonper.txt'
bc = 'periodic'

# NBody model object with parameters set above.
model = nb.NBody(m=m, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0, bc=bc, ndim=2)
=======
m = 1.0/npart

# NBody model object with parameters set above.
model = nb.NBody(m=m, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0, bc='grounded')

# import matplotlib.pyplot as plt
# plt.imshow(model.get_pot())
# plt.show()
>>>>>>> parent of 0f23b3a... first energy results

# Animation (.gif file)
niter = 400
title = (r'{} randomly scattered particles for $dt={}$ ({} frames)'
         ).format(npart,
                  dt,
                  niter)

ut.grid_animation2d(model, niter=niter, show=show, savepath=fpath,
<<<<<<< HEAD
                    figsize=None, intv=50, nsteps=1, title=title,
                    repeat=False, marker='r*', style='grid', logfile=logfile)
=======
                    figsize=None, intv=50, ret_fig=False, ret_ani=False,
                    nsteps=1, title=title, repeat=False, marker='r*',
                    style='grid')
>>>>>>> parent of 0f23b3a... first energy results
