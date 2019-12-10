import warnings

import numpy as np


class NBody():
    """Particle Mesh (PM) NBody solver

    Modelling gravitational N-Body problem with particle mesh method in 2D.
    Solving for the potential with Fourier methods and flexible mass/force
    assignment/interpolation.

    Args:
        m (array or float): Mass of the particles. Same mass for all particles
                            if scalar, one mass per ptcl if array.
        npart (int): number of particles
        ngrid (int): number of grid (mesh) pts/cells
        soft (float): softening constant applied to distance in potential/force
                      calculation
        dt (float): time steps
        pos0 (array): Initial position array with shape (npart, ndim).
                      Uniformly distributed if None, default is None.
                      If an array with values outside of ]0, ngrid), it is
                      automatically rescaled to fit the grid.
        vel0 (array or float): Initial velocity array with shape (npart, ndim).
                               If float, input times normal random number.
                               If None, automatically set to 0.
        G (float): gravitational constant
    Returns:
        nbody (NBody): NBody solver with specified conditions.
    """

    def __init__(self, m=1.0, npart=1000, ngrid=500, soft=0.1, dt=0.1,
                 pos0=None, vel0=None, G=1.0):
        # Constant parameters.
        self._npart = int(npart)
        if ngrid % 2 == 0:
            self._ngrid = int(ngrid)
        else:
            warnings.warn('Arg. ngrid must be even. Adding 1 to input',
                          RuntimeWarning)
            self._ngrid = int(ngrid) + 1
        if np.isscalar(soft):
            self._soft = soft
        else:
            raise TypeError('soft must be a scalar value')
        if np.isscalar(dt):
            self._dt = dt
        else:
            raise TypeError('dt must be a scalar value')
        if np.isscalar(G):
            self._G = G
        else:
            raise TypeError('G must be a scalar value')

        # Check and init mass in an (npart, 1) array. The shape is to
        # faciliate array operations with pos and vel.
        # Result stored in self._m
        self._init_mass(m)

        # Check and init ptcls position and velocity, both in (npart, 2) array.
        # Results stored in self._pos and self._vel
        self._init_pos(pos0)
        self._init_vel(vel0)

        # Initialize density (same call as all other density computations).
        # Result stored in self._density. Also assigns self._meshpts with same
        # shape as self._pos, with mesh points for each ptcl.
        self._compute_density()

        # Initialize Green's function for gravitational problem, with symmetry
        # to ensure periodicity of BCs. Result is stored in self._green
        self._init_green()

    # safe client-side access to some attributes
    @property
    def npart(self):
        return self._npart

    @property
    def ngrid(self):
        return self._ngrid

    @property
    def soft(self):
        return self._soft

    @property
    def dt(self):
        return self._dt

    @property
    def G(self):
        return self._G

    @property
    def m(self):
        return self._m

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def density(self):
        return self._density

    def _init_mass(self, m):
        if np.isscalar(m):
            self._m = m * np.ones([self.npart, 1], dtype=float)
        else:
            m = np.asarray(m, dtype=float)
            m = m.flatten()
            if m.size != self.npart:
                raise ValueError('m must have size npart or be scalar')
            self._m = np.array([m.copy()]).T

    def _init_pos(self, pos0):
        if pos0 is not None:
            # Sanity checks.
            pos = np.asarray(pos0, dtype=float)
            if pos.shape != (self.npart, 2):
                raise ValueError('pos0 must have shape (npart, 2)')
            if pos.max() >= self.ngrid or pos.min() < 0.0:
                raise ValueError('positions in all dimension must be within'
                                 ' [0, ngrid)')
        else:
            pos = self.ngrid
            pos *= np.random.random_sample([self.npart, 2])
        self._pos = pos.copy()

    def _init_vel(self, vel0):
        if vel0 is None:
            vel0 = 0.0
        if np.isscalar(vel0):
            vel = float(vel0)
            vel *= np.random.standard_normal([self.npart, 2])
        # Sanity checks.
        else:
            vel = np.asarray(vel0, dtype=float)
            if vel.shape != (self.npart, 2):
                raise ValueError('vel0 must have shape (npart, 2)')
        self._vel = vel.copy()

    def _compute_density(self):
        # Initialize density mesh grid. The index of each cell along a
        # dimension gives the position along this dimension.
        density = np.zeros([self.ngrid, self.ngrid], dtype=float)

        # Get mesh points for each ptcl. Take modulo to assign xi=L to xi=0,
        # such that mesh points can be integers in range [0, L-1].
        self._meshpts = np.rint(self.pos).astype('int') % self.ngrid

        # For each particle, assign density (mass bc unitary cells) to mesh
        # point.
        # NGP assigment scheme
        for i in range(self.npart):
            # Note: m[i] is an array but '+=' casts to float.
            density[tuple(self._meshpts[i])] += self.m[i]
        self._density = density

    def _init_green(self):
        # Compute Green's function on mesh grid.
        greensize = self.ngrid
        meshrange = np.arange(greensize, dtype=float)
        mesh = np.array(np.meshgrid(meshrange, meshrange))
        rsqr = np.sum(mesh**2, axis=0)
        rsqr[rsqr < self.soft**2] = self.soft**2
        rsqr += self.soft**2
        r = np.sqrt(rsqr)
        green = 1 / (4*np.pi*r)

        # Flip to get same behaviour in each corner (for periodicity).
        half = greensize // 2
        green[half:, :half] = np.flip(green[:half, :half], axis=0)
        green[:, half:] = np.flip(green[:, :half], axis=1)

        self._green = green

    def get_pot(self):
        """Compute model potential
        Get potential for each mesh cell by convoluting pre-calculated Green's
        function. Convolution is performed with 2 fourier transform.
        Returns:
            pot (array): potential at every mesh point.
        """
        # Perform convolution
        pot = np.fft.rfft2(self._green) * np.fft.rfft2(self.density)
        pot = np.fft.irfft2(pot)

        # Because of our periodic Green's function definition used in the FFT,
        # potential is not symmetric wrt ptcl positions, so we need to shift
        # and average to center it back on ptcls.
        for i in range(2):
            pot = 0.5 * (np.roll(pot, 1, axis=i) + pot)

        return pot

    def get_fmesh(self):
        """Compute mesh forces
        Get force components for each mesh point by differentiating current
        potential grid.
        Returns:
            fmesh (array): forces at each grid point. Array of shape
            (2, ngrid, ngrid) where first axis 0 is the x and y components.
        """
        # Init force array.
        fmesh = np.zeros([2, self.ngrid, self.ngrid], dtype=float)

        # Get pot.
        pot = self.get_pot()

        # periodic gradient in each direction
        for i in range(2):
            fmesh[i] = 0.5*(np.roll(pot, 1, axis=i) - np.roll(pot, -1, axis=i))

        # Times density and -1 to get actual forces
        fmesh *= - self.density

        return fmesh

    def get_fpart(self):
        """Compute particle forces
        Get force components for each particle by interpolating from mesh
        forces.
        Returns:
            fpart (array): forces on particle along each axis. Array has shape
                           (npart, 2).
        """
        # Init fpart array
        fpart = np.zeros([self.npart, 2])

        # Get mesh forces and transpose such that callig fmesh[i,j] will give
        # x and y components for mesh cell (i,j).
        fmesh = np.moveaxis(self.get_fmesh(), 0, -1)

        # Interpolate for each praticle using same scheme as in mass
        # assignment.
        # NGP interpolation.
        for i in range(self.npart):
            fpart[i] = fmesh[tuple(self._meshpts[i])]
        # fpart = fmesh[tuple(self._meshpts[:, i] for i in range(2))]

        return fpart

    def get_energy(self):
        """Get energy of the system
        Kinetic energy is calculated using ptcl velocities and masses.
        Potential energy is obtained with get_pot(), summing over all cells
        """
        kin = np.sum(self.m * self.vel**2)
        pot = 0.5*np.sum(self.density * self.get_pot())

        return kin + pot

    def evolve(self, nsteps=1):
        """Evolve sytem
        Let system evolve by a given number of timesteps. Leapfrog integration
        is used to solve for position and velocity.
        Args:
            nsteps (int): number of steps taken in given direction
        Returns:
            eg (float): total energy after letting the system evolve
        """
        for i in range(nsteps):
            # Get force for each particle
            forces = self.get_fpart()
            print(forces)

            # update velocity
            self._vel += forces * self.dt / self._m

            # update position
            self._pos += self.vel * self.dt
            self._pos = self._pos % self.ngrid

            # update grid
            self._compute_density()

        return self.get_energy()
