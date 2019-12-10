import numpy as np
import matplotlib.pyplot as plt

import nbody as nb
import utils as ut

npart = 1
ngrid = 50
pos0 = np.array([[ngrid/2, ngrid/2]])
# pos0 = np.array([[0, 0]])
# vel0 = 0.0
vel0 = np.array([[10, 0]])
model = nb.NBody(m=1.0, npart=1, ngrid=ngrid, soft=0.05, dt=0.1, pos0=pos0,
                 vel0=vel0, G=1.0)

# # initial density grid
# plt.imshow(model.density, origin='lower')
# plt.show()
# plt.imshow(model.get_pot(), origin='lower')
# plt.show()

ut.grid_animation2d(model, niter=2, show=False, savepath='./test.gif',
                    figsize=None, intv=50, ret_fig=False, ret_ani=False,
                    title=None, repeat=False)
