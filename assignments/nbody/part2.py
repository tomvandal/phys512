"""
Showing that two objects in circular orbit remain in circular orbit for some
time with NBody model.
"""
import numpy as np

import nbody as nb
import utils as ut

# Setting up two objects in orbit around center point.
npart = 2
ngrid = 400
pos0 = np.array([[ngrid/2, ngrid/2+10],
                 [ngrid/2, ngrid/2-10]])
vel0 = np.array([[1, 0],    # orbit: opposite velocities
                 [-1, 0]], dtype=float)
vfact = 0.1
vel0 *= vfact
soft = 0.1
dt = 5.0
fpath = './part2.gif'
show = False
m = 5.0

# NBody model object with parameters set above.
model = nb.NBody(m=5.0, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0)

# import matplotlib.pyplot as plt
# plt.imshow(model.get_pot())
# plt.show()

# Animation (.gif file)
niter = 100
title = (r'Orbiting bodies with $v_i={{{}}}$, $m={}$ for $dt={}$ ({} frames)'
         ).format(vfact,
                  m,
                  dt,
                  niter)

ut.grid_animation2d(model, niter=niter, show=show, savepath=fpath,
                    figsize=None, intv=50, ret_fig=False, ret_ani=False,
                    title=title, repeat=False, marker='r*', style='pts',
                    nsteps=1)
