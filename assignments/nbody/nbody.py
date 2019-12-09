"""
Definition of a class that simulates the N-body problem
"""
import warnings

import numpy as np


class NBody():
    """NBody simulator

    Modelling the gravitational NBody problem with particle mesh method and
    Fourier methods to solve the poisson equation.

    Args:
        m (array or float): Mass of the particles. Same mass for all particles
                            if int, one mass per ptcl if array.
        npart (int): number of particles
        resol (int): resolution (number of mesh cells) for all dimensions.
                     This must be even. If not, 1 is added to the input.
        soft (float): softening constant
        dt (float): time steps
        pos0 (array): Initial position array with shape (npart, ndim).
                      Uniformly distributed if None, default is None.
                      If an array with values greater than resol, it is
                      automatically rescaled to fit the grid.
        vel0 (array or float): Initial velocity array with shape (npart, ndim).
                               If float, input times normal random number.
                               If None, automatically set to 0.
        G (float): gravitational constant
        ndim (int): number of dimensions in the model
        bc (str): keyword specifying boundary condtions with the available
                  options.
                    - 'periodic': periodic bc for all dims (default).
                    - 'clamped': potential set to 0 at (padded) boundary.
    """

    def __init__(self, m=1.0, npart=1000, resol=1000, soft=0.5, dt=0.1,
                 pos0=None, vel0=None, G=1.0, ndim=2, bc='periodic'):
        # assign constants
        self._soft = soft
        self._npart = npart
        self._G = G
        self._dt = dt
        self._ndim = ndim
        self._npad = 5

        # boundary conditions
        bc_opts = ['periodic', 'clamped']
        if bc not in bc_opts:
            raise ValueError('bc must be one {} or {}'.format(*bc_opts))
        self._bc = bc

        # mesh resolution
        if resol % 2 == 0:
            self._resol = resol
        else:
            warnings.warn('resol must be even. Adding 1 to odd input.',
                          RuntimeWarning)
            self._resol = resol + 1

        # masses (with shape matching pos, vel)
        if np.isscalar(m):
            self._m = m * np.ones([npart, 1])
        else:
            m = np.asarray(m)
            m = m.flatten()
            if m.size != npart:
                raise ValueError('m must have size npart or be scalar')
            self._m = np.array([m.copy()]).T
        self._m = self._m.astype('float')  # ensure floats

        # initial position
        if pos0 is not None:  # if given, check and assign
            pos0 = np.asarray(pos0)
            if pos0.shape != (self.npart, self.ndim):
                raise ValueError('pos0 must have shape (npart, ndim)')
            self._pos = pos0.copy()
        self._init_pos()  # initialize properly (in all cases)

        # initial velocity
        if vel0 is None:
            vel0 = np.zeros([self.npart, self.ndim])
        else:
            if np.isscalar(vel0):
                vel0 *= np.random.standard_normal([self.npart, self.ndim])
            else:
                vel0 = np.asarray(vel0)
                if vel0.shape != (self.npart, self.ndim):
                    raise ValueError('vel0 must have shape (npart, ndim)')
        self._vel = vel0.copy()

        # Initial density mesh (and cell central pts)
        self._compute_density()

        # Evaluate Green's function on grid (need to do once only)
        self._init_green()

    @property
    def soft(self):
        return self._soft

    @property
    def npart(self):
        return self._npart

    @property
    def ndim(self):
        return self._ndim

    @property
    def pos(self):
        return self._pos

    @property
    def resol(self):
        return self._resol

    @property
    def vel(self):
        return self._vel

    @property
    def density(self):
        return self._density

    @property
    def bc(self):
        return self._bc

    def _init_pos(self):
        """Initialize position array
        If pos array was already assigned, simply rescales to [0, resol).
        Otherwise, creates pos array with random uniform distribution.
        """
        # Try as if already assigned, otherwise caught by 'except'
        try:
            # check if fits bounds and rescale if needed
            if np.any(np.logical_or(self.pos < 0, self.pos >= self._resol)):
                # rescale to resolution range
                self._pos -= self.pos.min()
                self._pos /= (self.pos.max() - self.pos.min() + 1)
                self._pos *= self._resol
        except AttributeError:
            self._pos = self._resol
            self._pos *= np.random.random_sample([self.npart, self.ndim])

    def _compute_density(self):
        """Compute mesh density for current position
        Updates grid density using position and interpolation method sepecified
        at initialization.
        Result is assigned to self.density
        """
        # new grid
        self._density = np.zeros([self._resol]*self._ndim, dtype='float')

        # assing indices in x and y by rounding down and convert to int
        self._indxy = np.floor(self._pos).astype('int')

        # assign densities according to interpolation scheme
        for i in range(self._npart):
            self._density[tuple(self._indxy[i])] += self._m[i, 0]

    def _init_green(self):
        """Compute Green's function on the grid
        This needs to run once at initialization since it is related to cell
        positions and not particles. In the case of gravitational foce problem,
        Green's function will be 1/(4*pi*r). We assign it in first 1/2^ndim of
        the ndim grid and then flip it around to get periodic behaviour in the
        space.
        """
        # first 1/2^ndim of the space: evaluate Green's funct.
        inds = np.repeat([np.arange(self._resol//2)], self._ndim, axis=0)
        indmesh = np.array(np.meshgrid(*inds))
        indmesh = indmesh.astype('float')
        # r = np.sqrt(np.sum(indmesh**2, axis=0))  # eval dists from corner
        # r[r < self._soft] = self._soft  # override small dists with soft
        # r += self._soft  # add soft everywhere for consistency
        soft = self._soft**2
        rsqr = np.sum(indmesh**2, axis=0)
        rsqr[rsqr < soft] = soft
        rsqr += soft
        r = np.sqrt(rsqr)
        green = 1.0 / (4*np.pi*r)

        # flip along each dimension to get final array
        # NOT TESTED IN 3D YET
        for i in range(self._ndim):
            green = np.append(green, np.flip(green, axis=i), axis=i)

        # assign green function (evaluated on grid) to attribute
        self._green = green

    def _compute_pot(self):
        """Compute potential
        Gives potential in each grid cell. These potential values do not
        account for the mass. The potential array is padded according to the
        boundary condtion.
        Returns:
            pot (array): array representing the potential grid, with BC padding
                         such that each axis has size resol+2
        """
        pot = np.fft.rfftn(self._green) * np.fft.rfftn(self._density)
        pot = np.fft.irfftn(pot)

        # apply BC
        if self._bc == 'periodic':
            pot = np.pad(pot, self._npad, mode='wrap')  # wrap each dim on itself
        elif self._bc == 'clamped':
            pot = np.pad(pot, self._npad, constant_values=0.0)

        # re-center potential using adjacent cells, improving stability
        for i in range(self._ndim):
            pot = 0.5 * (np.roll(pot, 1, axis=i) + pot)

        return pot

    def get_pot(self):
        """Get potential
        Gives potential in each grid cell. These potential values do not
        account for the mass.
        Returns:
            pot (array): array representing the potential grid
        """
        pot = self._compute_pot()
        pot = NBody._strip(pot, nlayers=self._npad)  # remove padding

        return pot

    @staticmethod
    def _strip(a, nlayers=1, keep=None):
        """Strip an array with arbitrary number of dimension
        Along each dimension, given number of layers is stripped from the input
        arra.
        Args:
            a (array): array with arbitrary number of dimension.
            nlayers (int): number of layers to strip along each dimension
            keep (array_like): dimensions along which we keep all elements.
                               Default is None.
        Returns:
            astrip (array): copy of a with nlayers removed axes
        """
        # process input
        nlayers = int(nlayers)
        a = np.array(a)
        a = a.copy()
        mask = np.ones(a.ndim, dtype=bool)  # dims along which we strip
        if keep is not None:
            if np.isscalar(keep):
                keep = [keep]
            keep = tuple(keep)
            mask[keep] = 0

        # indexing
        keepind = slice(None)
        stripind = slice(nlayers, -nlayers)
        inds = np.empty(a.ndim, dtype=object)
        inds[mask] = stripind
        inds[np.invert(mask)] = keepind
        inds = tuple(inds)

        return a[inds]

    def grid_forces(self):
        """Compute forces on the grid
        Gives force acting on each grid cell, with additional padding at
        boundaries.
        Returns:
            fgrid (array): array of the same shape as the density grid with
                           three forces component for each grid cell (x,y,z).
                           E.g.: to access pt (0,5) on grid, simply use
                           fgrid[0, 5].
        """
        pot = self._compute_pot()  # pot with padding at boundaries

        # differentiate to get forces
        fgrid = np.array(np.gradient(pot))

        # convert to actual force, pad density, but no ptcls allowed in padding
        fgrid *= - np.pad(self._density, self._npad)

        return fgrid.T

    def ptcl_forces(self):
        """Calculate force for each ptcl position
        Gives force in each dimension by interpolating from grid forces with
        interpolation NGP interpolation
        Returns:
            fpos (array): array of shape (npart, ndim) with forces components
                          in all dimension for each particle.
        """
        # get force in grid cells
        fgrid = self.grid_forces()

        # strip boundary padding (no ptcls there)
        fgrid = NBody._strip(fgrid, nlayers=self._npad, keep=2)

        # get ptcl force with 0th order interp (NGP)
        fpos = fgrid[tuple(self._indxy[:, i] for i in range(self._ndim))]

        return fpos

    def get_energy(self, tot=True):
        """Get total energy of the current state
        Args:
            tot (bool): returns total energy if true, otherwise pot E and kin E
                        as a tuple.
        Returns:
            If tot is True:
                eg (float): total energy
            If tot is False:
                (pot, kin): potential and kinetic energy as a tuple of floats.
        """
        # Get total energy
        # Note: get_pot() gives potential (not pot. energy), so we need to
        # multiply by masses (densities) on the grid, and by G do get
        # proper units (if specified).
        pot_eg = np.sum(self._density * self.get_pot()) * self._G
        kin_eg = np.sum(self._m * self._vel**2)

        return pot_eg + kin_eg if tot else (pot_eg, kin_eg)

    def evolve(self):
        """Evolve NBody system by one timestep
        Uses leapfrog algorithm to solve for next position and velocity
        (assuming they are offset by half a step).
        Returns total energy of the system after the step
        """
        # update position
        self._pos += self._vel * self._dt

        # Periodic bc: reappear on the other side
        self._pos = self._pos % self._resol  # periodic boundary

        # get forces for each ptcl
        forces = self.ptcl_forces()

        # update velocity (have to play with m shape a bit to divide each ptcl)
        self._vel += forces * self._dt / self._m  # acc*dt

        # update grid
        self._compute_density()

        # Get total energy (after grid was updated in new state)
        return self.get_energy(tot=True)
