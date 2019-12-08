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

    def get_pot(self, pad=False):
        """Compute potential energy
        Gives potential energy in each grid cell (not for each ptcl!). These
        potential values do not account for the mass
        Args:
            pad (bool): return with padding for clamped bc if true
        Returns:
            pot (array): array representing the potential grid
        """
        pot = np.fft.rfftn(self._green) * np.fft.rfftn(self._density)
        pot = np.fft.irfftn(pot)

        # re-center potential using adjacent cells, improving stability
        for i in range(self._ndim):
            pot = 0.5 * (np.roll(pot, 1, axis=i) + pot)

        # handle non-periodic boundaries
        if self.bc == 'clamped' and pad:
            pot = np.pad(pot, 1, constant_values=0)

        return pot

    def grid_forces(self):
        """Compute forces on the grid
        Gives force acting on each grid cell.
        Returns:
            fgrid (array): array of the same shape as the density grid with
                           three forces component for each grid cell (x,y,z).
                           E.g.: to access pt (0,5) on grid, simply use
                           fgrid[0, 5].
        """
        pot = self.get_pot(pad=True)  # will be padded if clamped bc
        if self.bc == 'periodic':
            shape = [self._ndim]
            shape.extend([self._resol]*self._ndim)
            fgrid = np.zeros(shape)
            for i in range(self._ndim):
                fgrid[i] = -0.5 * (np.roll(pot, 1, axis=i)
                                   - np.roll(pot, -1, axis=i))
        if self.bc == 'clamped':
            shape = [self._ndim]
            shape.extend([self._resol+2]*self._ndim)
            fgrid = np.zeros(shape)
            for i in range(self._ndim):
                fgrid[i] = -0.5 * (np.roll(pot, 1, axis=i)
                                   - np.roll(pot, -1, axis=i))
            # fgrid = np.array(np.gradient(pot))  # get grad of pot, transposed
            inds = [slice(None)]
            inds.extend([slice(1, -1)]*self._ndim)
            inds = tuple(inds)
            fgrid = fgrid[inds]
            # USE PADDING ALL THE TIME INSTEAD OF WEIRD PC STUFF
        fgrid *= - self._density  # convert to actual force

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
