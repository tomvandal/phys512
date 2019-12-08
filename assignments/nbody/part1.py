"""
Part 1: Showing that a single particle at rest remains at rest remains
motionless.
"""
import numpy as np
import matplotlib.pyplot as plt

import nbody as nb
import utils as ut

# init nbody model
res = 1000
# pos0 = np.array([[res / 2, res / 2]])
# pos0 = np.array([[res / 2, res / 2],
#                  [30, 30]])
pos0=None
# vel0 = np.array([[1.0, 0]])
vel0 = 0.0
model = nb.NBody(m=1.0, npart=1000, resol=res, soft=0.01, dt=1.0, pos0=pos0,
                 vel0=vel0, G=1.0, ndim=2, bc='periodic')
# print(model.pos)
# print(model.vel)
# print(model.get_energy())
# for i in range(50):
#     print(i)
#     model.evolve()
#     plt.imshow(model.density)
#     plt.pause(0.01)
#     print(model.pos)
#     print(model.vel)
#     print(model.get_energy())
# model.evolve()

# show model
ut.make_animation2d(model, niter=200, show=True, savepath=None, figsize=(8, 8),
                    intv=200, ret_fig=False, ret_ani=False, title=None,
                    repeat=True)