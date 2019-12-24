"""
Simulating periodic BC universe with scale-invariant power spectrum s.t. mass
fluctuations are proportional to k^(-3).
"""
import nbody as nb
import utils as ut

# Setting up two objects in orbit around center point.
npart = 2**17
ngrid = 2**9
pos0 = None
vel0 = 0.0
soft = 10
dt = 100.0
fpath = None  # './fig4.gif'
show = True
m = 1.0/npart  # scale mass with npart to limit total energy
logfile = None  # './energy4.txt'

# NBody model object with parameters set above.
model = nb.NBody(m=m, npart=npart, ngrid=ngrid, soft=soft, dt=dt, pos0=pos0,
                 vel0=vel0, G=1.0, bc='periodic', cosmo=True)

# Animation (.gif file)
niter = 400
title = (r'{} randomly scattered particles for $dt={}$ ({} frames)'
         ).format(npart,
                  dt,
                  niter)

ut.grid_animation2d(model, niter=niter, show=show, savepath=fpath,
                    figsize=None, intv=50, nsteps=1, title=title, repeat=False,
                    marker='r*', style='grid', norm=None, cmap=None,
                    logfile=logfile)
