"""
Simulating time-evolution of hundreds of thousands of particles randomly
scattered with NBody model. Boundary conditions are passed as an argument to
NBody object. Both periodic and non-periodic BCs are implemented.
"""
import numpy as np

import nbody as nb
import utils as ut

# Setting up two objects in orbit around center point.
npart = int(1e5)
ngrid = 400
pos0 = None
vel0 = 0.0
soft = 1
dt = 100.0
fpath = None  # './part3.gif'
show = True
m = 1.0/npart

# NBody model object with parameters set above.
model = nb.NBody(m=m, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0)

# import matplotlib.pyplot as plt
# plt.imshow(model.get_pot())
# plt.show()

# Animation (.gif file)
niter = 400
title = (r'{} randomly scattered particles for $dt={}$ ({} frames)'
         ).format(npart,
                  dt,
                  niter)

ut.grid_animation2d(model, niter=niter, show=show, savepath=fpath,
                    figsize=None, intv=50, ret_fig=False, ret_ani=False,
                    nsteps=1, title=title, repeat=False, marker='r*',
                    style='grid')
