"""
Class for a 2d box containing a cylinder. Some private methods useful
for the various solvers are defined before the Box class.
"""
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# nice plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def _get_rhs(bc, mask):
        """RHS of the matrix formulation
        Get RHS (b) of the problem in matrix formulation Ax=b. This gives the
        contribution from the boundary conditions.
        Args:
            bc   (array): potential boundary conditions
            mask (array): where the boundary conditions apply in theh potential
                          matrix
        Returns:
            rhs (array): matrix of the BC contribution to final potential
        """
        rhs = np.zeros(bc.shape)
        rhs[:, :-1] += bc[:, 1:]  # add right
        rhs[:, 1:] += bc[:, :-1]  # add left
        rhs[:-1, :] += bc[1:, :]  # add down
        rhs[1:, :] += bc[:-1, :]  # add up
        rhs[mask] = 0  # where BC hold, just put 0, handled in other methods

        return rhs


def _get_laplacian(pot, mask, copy=False):
    """Get laplacian operator with finite differences.
    Args:
        pot  (array): potential grid
        mask (array): True where BCs apply
    Returns:
        ax (array): laplacian opereator in matrix form
    """
    if copy:
        pot = pot.copy()
    pot[mask] = 0  # drop boundaries
    ax = 4 * pot  # get rid of *4 in eq
    ax[:, :-1] -= pot[:, 1:]  # subtract right
    ax[:, 1:] -= pot[:, :-1]  # subtract left
    ax[:-1, :] -= pot[1:, :]  # subtract down
    ax[1:, :] -= pot[:-1, :]  # subtract up
    ax[mask] = 0  # put 0 where BC hold, handled in other methods

    return ax


def _do_cg(bc, mask, pot0=None, maxit=None, tol=1e-2):
    """Poisson CG solver
    Poisson solver with conjugate gradient method
    Args:
        bc   (array): potential boundary conditions
        mask (array): bool array where bc applies
        pot0 (array): Initial potential array
        maxit  (int): Max number of iterations
        tol  (float): Maximum chi2 for convergence
    Returns:
        pot (array): final potential grid with bc enforced
    """
    npix = bc.shape[0]
    if maxit is None:
        maxit = npix
    if pot0 is None:
        pot0 = bc.copy()
    rhs = _get_rhs(bc, mask)  # reference from BC
    ax = _get_laplacian(pot0, mask)
    r = rhs - ax
    p = r.copy()
    pot = pot0.copy()
    rtr = np.sum(r*r)
    for i in range(maxit):
        ap = _get_laplacian(p, mask)
        alpha = rtr / np.sum(ap*p)
        pot += alpha * p
        r -= alpha * ap
        rtr_new = np.sum(r*r)
        beta = rtr_new / rtr
        p = r + beta * p
        rtr = rtr_new
        if rtr < tol:
            msg = ("The CG method converged after {} "
                   "iterations.").format(i)
            print(msg)
            break
        elif i == maxit-1:
            msg = ("The maximum of {} iterations was reached before"
                   "convergence... potential may be poorly constrained."
                   ).format(maxit)
            warnings.warn(msg, RuntimeWarning)
            break
    pot[mask] = bc[mask]  # enforce bc

    return pot, i


def _lowres(mat):
    """Lower resolution of matrix
    Lower the resolution of an array by a factor of 2 along each dimension.
    Take the maximum in absolute value, so that mask is conserved and BC
    is too. This method is not intended for other arrays than BC or masks.
    Args:
        mat (array): 2d array with equal even dimensions along both axes
    Returns:
        newmat (array): lower resolution matrix
    """
    # sanity check
    s = mat.shape
    assert np.unique(s).size == 1, 'mat must have equal dimensions'
    assert s[0] % 2 == 0, 'mat must have even dimensions'

    # split in 2x2 blocks and take abs max of each, then reshape
    n = s[0]
    newmat = mat.reshape(n//2, 2, -1, 2).swapaxes(1, 2).reshape(-1, 2, 2)
    absmax = np.max(np.abs(newmat), axis=(1, 2))
    truemax = np.max(newmat, axis=(1, 2))
    truemin = np.min(newmat, axis=(1, 2))
    newmat = np.where(np.equal(absmax, truemax), truemax, truemin)
    newmat = newmat.reshape(n//2, n//2)

    return newmat


def _upres(mat):
    """Increase matrix resolution
    Increase the resolution of an array by a factor of 2. This will only be
    used to initialize potential on a grid, so we don't need to care about
    precise BC/mask. We still use interpolation to make convergence faster.
    Args:
        mat (array): 2d array with equal dimensions along both axes
    Returns:
        newmat (array): array with higher resolution
    """
    # sanity check
    s = mat.shape
    assert np.unique(s).size == 1, 'mat must have equal dimensions'

    # get new interpolated grid with better resolution
    n = s[0]
    sizearr = np.linspace(0, n-1, num=n)
    xx, yy = np.meshgrid(sizearr, sizearr)
    pts = np.array([xx.ravel(), yy.ravel()]).T
    sizearr = np.linspace(0, n-1, num=2*n)
    xx, yy = np.meshgrid(sizearr, sizearr)
    ipts = np.array([xx.ravel(), yy.ravel()]).T
    newmat = griddata(pts, mat.ravel(), ipts, method='cubic')
    newmat = newmat.reshape(2*n, 2*n)

    return newmat


class Box:
    """ Cubic box with cylindrical conductor
    Cubic box containing a charged cylindrical conductor with constant
    potential. The box is represented as a square and the cylinder as a circle
    in a 2D x-y plane to lighten computations. This is still consistent with a
    3D cylinder since the behavior is the same everywhere along z.

    Args:
        npix       (int): number of pixel corresponding to the desired resolution
                          along each dimension
        radius     (int): radius of the cylinder (in pixels)
        cylpot   (float): constant potential inside the cylinder (in volts)
        edge_bc    (str): boundary conditions at edges of the box
                          - 'clamped': 0 everywhere
        bumploc  (float): angular location of the bump's center on the cylinder
                          (no bump if None, default is None)
        bumpfrac (float): ratio of bump radius to wire radius
                          (ignored if bumploc is None)
    """

    def __init__(self, npix=512, radius=30, cylpot=2, edge_bc='clamped',
                 bumploc=None, bumpfrac=None):

        # Define charcteristics.
        self._npix = npix
        self._radius = radius
        self._cylpot = cylpot

        # BCs at edges
        self._bc = np.zeros([self._npix, self._npix])  # zero OK for clamped

        # grid for evaluations
        xyarr = np.arange(-self._npix/2, self._npix/2)
        self._xx, self._yy = np.meshgrid(xyarr, xyarr)  # coords

        # Mask where BC applied (edges for now)
        self._mask = np.zeros([self._npix, self._npix], dtype='bool')
        self._mask[0, :] = True
        self._mask[-1, :] = True
        self._mask[:, 0] = True
        self._mask[:, -1] = True

        # add cylinder at center
        cylcond = self._xx**2 + self._yy**2 <= self._radius**2  # inside cyl
        self._bc[cylcond] = self._cylpot
        self._mask[cylcond] = True

        # initial potential (same as BC)
        self._pot = self._bc.copy()

        # add bump if specified
        if bumploc is not None:
            xb = self._radius * np.cos(bumploc)
            yb = self._radius * np.sin(bumploc)
            self._bradius = bumpfrac * self._radius

            # update bc and mask
            bcond = (self._xx - xb)**2 + (self._yy - yb)**2 <= self._bradius**2
            self._bc[bcond] = self._cylpot
            self._mask[bcond] = True
        else:
            self._bradius = None

    # A few property to access the box parameters without being able to modify
    # them.
    @property
    def npix(self):
        return self._npix

    @property
    def radius(self):
        return self._radius

    @property
    def cylpot(self):
        return self._cylpot

    @property
    def pot(self):
        return self._pot

    def addbump(self, bumploc, bumpfrac, hard=False):
        """Add a bump on the cylinder
        There is a limit of 1 bump.
        Args:
            bumploc  (float): angular location of the bump's center on the
                              cylinder (no bump if None, default is None)
            bumpfrac (float): ratio of bump radius to wire radius
                              (ignored if bumploc is None)
            hard      (bool): if true, an error is raised when the bump is
                              already defined. Otherwise, only a warning is
                              raised (default is false)
        """
        # check there is no bump
        if self._bradius is not None:
            if hard:
                raise RuntimeError('The limit of 1 bump has already been'
                                   ' reached.')
            else:
                msg = ('The limit of 1 bump has already been reached. No new'
                       ' bump will be added.')
                warnings.warn(msg, RuntimeWarning)
                return

        # add a bump
        xb = self._radius * np.cos(bumploc)
        yb = self._radius * np.sin(bumploc)
        self._bradius = bumpfrac * self._radius

        # update bc and mask
        bcond = (self._xx - xb)**2 + (self._yy - yb)**2 <= self._bradius**2
        self._bc[bcond] = self._cylpot
        self._mask[bcond] = True

    def relax(self, maxit=None, tol=1e-2):
        """Relaxation method solver
        Use a simple, non-optimized relaxation method to solve for the
        potential everyhwere in the box.
        Args:
            maxit (int): Max number of iterations
            tol (float): Maximum chi2 for convergence
        """
        start = time.time()
        if maxit is None:
            maxit = self.npix
        # start with BC
        self._pot = self._bc.copy()

        rhs = _get_rhs(self._bc, self._mask)  # 'true' reference
        ax = _get_laplacian(self._pot, self._mask, copy=True)  # first lapl.
        r = rhs - ax  # first residuals
        rtr = np.sum(r*r)  # first chi2 (unit error)
        for i in range(maxit):
            self._pot[1:-1, 1:-1] = (self._pot[1:-1, :-2]
                                     + self._pot[1:-1, 2:]
                                     + self._pot[:-2, 1:-1]
                                     + self._pot[2:, 1:-1]) / 4.0
            self._pot[self._mask] = self._bc[self._mask]  # enforce BC
            r = rhs - _get_laplacian(self._pot, self._mask, copy=True)
            rtr = np.sum(r*r)
            if rtr < tol:
                ttime = time.time() - start
                msg = ("The relaxation method converged after {} "
                       "iterations in {} seconds.").format(i, ttime)
                print(msg)
                break
            elif i == maxit-1:
                ttime = time.time() - start
                msg = ("The maximum of {} iterations was reached before"
                       " convergence, in {} seconds. Potential may be poorly"
                       " constrained.").format(maxit, ttime)
                warnings.warn(msg, RuntimeWarning)
                break

    def cg(self, pot0=None, maxit=None, tol=1e-2):
        """Poisson CG solver
        Poisson solver with conjugate gradient method
        Args:
            pot0 (array): Initial potential array
            maxit  (int): Max number of iterations
            tol  (float): Maximum chi2 for convergence
        """
        start = time.time()
        self._pot, count = _do_cg(self._bc, self._mask, pot0=pot0, maxit=maxit,
                                  tol=tol)
        ttime = time.time() - start
        print('CG took {} iterations, {} seconds'.format(count, ttime))

    def cgres(self, pot0=None, nres=6, maxit=None, tol=1e-2):
        """Poisson CG solver with resolution loop
        Poisson solver with conjugate gradient method looping over resolution
        to facilitate convergence.
        Args:
            pot0 (array): Initial potential array
            nres   (int): Number of resultions (dividing by 2 each time)
            maxit  (int): Max number of iterations
            tol  (float): Maximum chi2 for convergence
        """
        start = time.time()
        # initialize lists for all res
        all_mask = [None] * nres
        all_bc = [None] * nres
        all_pot = [None] * nres
        all_mask[0] = self._mask
        all_bc[0] = self._bc
        for i in range(1, nres):
            all_mask[i] = _lowres(all_mask[i-1])
            all_bc[i] = _lowres(all_bc[i-1])

        # base case (lowest res)
        all_pot[-1], count = _do_cg(all_bc[-1], all_mask[-1], pot0=pot0,
                                    maxit=maxit, tol=tol)

        # downgrade resolution nres-1 times from base case (lowest)
        for i in range(nres-2, -1, -1):
            pot0 = _upres(all_pot[i+1])
            all_pot[i], counti = _do_cg(all_bc[i], all_mask[i], pot0=pot0,
                                        maxit=maxit, tol=tol)
            count += counti
        ttime = time.time() - start
        print('CG took {} total iterations, {} seconds'.format(count, ttime))

        self._pot = all_pot[0]  # update with best (desired) res

    def get_rho(self):
        """Charge density
        Calculate the charge density everywhere in the box
        Returns:
            rho (array): charge density distribution in the box
        """
        # poisson equation gives density directly
        rho = _get_laplacian(self._pot, self._mask, copy=True)[1:-1, 1:-1]

        return rho

    def get_efield(self, sparse=None, theo=False):
        """Get Electric field
        Args:
            sparse (float): resampling frequency, all pts if None
            theo   (float): calculate analytic E-field if true (deault: false)
        Returns:
            ex, ey: x and y Efields
        """
        if sparse is None:
                sparse = 1
        if not theo:
            ey, ex = np.gradient(-self.pot[::sparse, ::sparse])
        else:
            ey, ex = np.gradient(-self.get_vtheo()[::sparse, ::sparse])

        return ex, ey

    def get_vtheo(self):
        """Theoretical potential
        Calculate Theoretical potential based on cylinder V and BC
        Returns:
            v (array): analytic potential across array
        """
        dv = self._bc[0, 0] - self._cylpot
        logr = np.log(np.sqrt(self._xx**2 + self._yy**2))
        lam = dv / (logr[0, 0] - logr[self.npix//2, self.npix//2+self.radius])
        const = self._cylpot - lam*logr[self.npix//2, self.npix//2+self.radius]
        v = lam*logr + const
        v[self._mask] = self._bc[self._mask]

        return v

    def show_threepanels(self, sparse=10, figsize=(12, 3)):
        """3 panel plot
        3 panel plot showing V, E on one panel, their analytic equivalent on
        on the other, and the charge density on the last one.
        Args:
            sparse (float): resampling spacing for E-field
            figsize (tuple): figure size according to plt.subplots
        """
        ex, ey = self.get_efield(sparse=sparse)

        labsize = 16
        titlesize = 20

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        vtheo = self.get_vtheo()
        extheo, eytheo = self.get_efield(sparse=sparse, theo=True)

        potcol = axs[0].pcolormesh(self._xx, self._yy, self.pot)
        axs[0].quiver(self._xx[::sparse, ::sparse],
                      self._yy[::sparse, ::sparse], ex, ey, angles='xy')
        axs[0].set_xlabel('X (pixels)', fontsize=labsize)
        axs[0].set_ylabel('Y (pixels)', fontsize=labsize)
        axs[0].set_title('Simulated Potential V', fontsize=titlesize)
        fig.colorbar(potcol, ax=axs[0])

        truepot = axs[1].pcolormesh(self._xx, self._yy, vtheo)
        axs[1].quiver(self._xx[::sparse, ::sparse],
                      self._yy[::sparse, ::sparse], extheo, eytheo,
                      angles='xy')
        axs[1].set_xlabel('X (pixels)', fontsize=labsize)
        axs[1].set_ylabel('Y (pixels)', fontsize=labsize)
        axs[1].set_title('Theoretical Potential V', fontsize=titlesize)
        fig.colorbar(truepot, ax=axs[1])

        charge = axs[2].pcolormesh(self.get_rho())
        axs[2].set_xlabel('X (pixels)', fontsize=labsize)
        axs[2].set_ylabel('Y (pixels)', fontsize=labsize)
        axs[2].set_title(r'Simulated Charge Density $\rho$',
                         fontsize=titlesize)
        fig.colorbar(charge, ax=axs[2])
        plt.tight_layout()
        plt.show()
