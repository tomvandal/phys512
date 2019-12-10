"""
Showing that a single particle at rest remains motionless with NBody model
"""
import numpy as np

import nbody as nb
import utils as ut

# Setting up the model.
npart = 1
ngrid = 2**6
pos0 = np.array([[ngrid/2, ngrid/2]])  # works anywhere, nicer plot centered
vel0 = 0.0                             # motionless ptcl
soft = 0.01
dt = 0.1
figpath = './part1.gif'
logfile = './energy1.txt'

niter = 50
title = 'Single Particle at rest with dt={} ({} frames)'.format(dt, niter)

# NBody model object with parameters set above.
model = nb.NBody(m=1.0, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0)

# Animation (.gif file)
ut.grid_animation2d(model, niter=niter, show=False, savepath=figpath,
                    figsize=None, intv=200, title=title, repeat=False,
                    marker='ko', style='pts', logfile=logfile)
