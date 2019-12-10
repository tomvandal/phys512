import warnings

import numpy as np

BC_OPTS = ['periodic', 'grounded']  # available BC options


class NBody():
    """Particle Mesh (PM) NBody solver

    Modelling gravitational N-Body problem with particle mesh method in n-D.
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
        bc (str): Keyword specifying how to deal with boundary conditions.
                  Options are:
                    - None: do nothing when a particle reaches boundaries
                            (default). NOT RECOMMENDED.
                    - 'periodic': periodic boundary condtions. The particles
                                  reappear on the other side of the space.
                    - 'grounded': potential is set to 0 around the boundary.
    Returns:
        nbody (NBody): NBody solver with specified conditions.
    """

    def __init__(self, m=1.0, npart=1000, ngrid=500, soft=0.1, dt=0.1,
                 pos0=None, vel0=None, G=1.0, bc='periodic', cosmo=False,
                 ndim=2):
        # Constant parameters.
        # Number of ptcls
        self._npart = int(npart)
        # Grid pts
        if ngrid % 2 == 0:
            self._ngrid = int(ngrid)
        else:
            warnings.warn('Arg. ngrid must be even. Adding 1 to input',
                          RuntimeWarning)
            self._ngrid = int(ngrid) + 1
        # Softening constant.
        if np.isscalar(soft):
            self._soft = soft
        else:
            raise TypeError('soft must be a scalar value')
        # Time steps.
        if np.isscalar(dt):
            self._dt = dt
        else:
            raise TypeError('dt must be a scalar value')
        # Grav. constant
        if np.isscalar(G):
            self._G = G
        else:
            raise TypeError('G must be a scalar value')
        # BCs
        if bc not in BC_OPTS:
            raise ValueError('bc must be one of {}'.format(BC_OPTS))
        self._bc = bc
        if isinstance(cosmo, bool):
            self._cosmo = cosmo
        else:
            raise TypeError('cosmo must be boolean.')
        self._ndim = int(ndim)

        # Check and init ptcls position and velocity, both in (npart, ndim)
        # array. Results stored in self._pos and self._vel
        self._init_pos(pos0)
        self._init_vel(vel0)

        # Check and init mass in an (npart, 1) array. The shape is to
        # faciliate array operations with pos and vel.
        # Result stored in self._m
        self._init_mass(m)

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
    def ndim(self):
        return self._ndim

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
    def bc(self):
        return self._bc

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

    @property
    def cosmo(self):
        return self._cosmo

    def _init_mass(self, m):
        if not self.cosmo:
            # Sanity check and simple array manipulation.
            if np.isscalar(m):
                self._m = m * np.ones([self.npart, 1], dtype=float)
            else:
                m = np.asarray(m, dtype=float)
                m = m.flatten()
                if m.size != self.npart:
                    raise ValueError('m must have size npart or be scalar')
                self._m = np.array([m.copy()]).T
        # Using scale-invariant power spectrum to scale mass distribution. In
        # this case, the grid points will be used to obtain nd spacial
        # frequencies (k-vector), and this will be used to obtain a power
        # spectrum. We will then re-scale mass distribution and assign scaled
        # masses to each particle.
        # (https://cds.cern.ch/record/583256/files/0209590.pdf)
        else:
            # First, need to uniformize masses for all ptcls
            if not np.isscalar(m):
                msg = ('m must be scalar when cosmo is True. Assigning '
                       'm to mean of input array')
                warnings.warn(msg, RuntimeWarning)
                m = float(np.mean(m))
            self._m = m * np.ones([self.npart]).T

            # Compute density grid
            self._compute_density()
            mavg = np.mean(self._m)
            davg = np.mean(self.density)

            # Fourier space and power spectrum.
            knd = np.fft.rfftfreq(self.ngrid) * 2*np.pi
            knd = np.repeat([knd], self.ndim, axis=0)
            kmesh = np.array(np.meshgrid(*knd))  # mesh in all dims
            ksqr = np.sum(kmesh**2, axis=0)  # k2 on grid
            soft = 2*np.pi/self.soft
            ksqr[ksqr < soft**2] = soft**2
            ksqr += soft**2
            k = np.sqrt(ksqr)
            powerspec = 1.0 / k**3

            # Draw params for each fluctuation mode (each k), and construct FT.
            amps = np.random.rayleigh(scale=np.sqrt(powerspec))
            phases = 2*np.pi*np.random.random_sample(amps.shape)
            ft = amps * np.exp(1j*phases)

            # IFT for fluctuations, then derive ptcl masses
            fluct = np.fft.irfftn(ft, s=self.density.shape)
            density = davg * fluct + davg  # new density
            ncell = self.density / mavg  # ptcls in each cell
            mcell = density.copy()
            mcell[ncell > 0] /= ncell[ncell > 0]  # mass for ptcls in each cell

            # Assign back to mass.
            m = mcell[self._meshinds]

            self._m = np.array([m.copy()]).T

    def _init_pos(self, pos0):
        if self.bc == 'periodic':
            xmin, xmax = 0, self.ngrid
        else:
            xmin, xmax = 1, self.ngrid - 1
        if pos0 is not None:
            # Sanity checks.
            pos = np.asarray(pos0, dtype=float)
            if pos.shape != (self.npart, self.ndim):
                raise ValueError('pos0 must have shape (npart, ndim)')
            if pos.max() >= xmax or pos.min() < xmin:
                raise ValueError('positions in all dimension must be within'
                                 ' [0, ngrid) for periodic and [1,ngrid-1)'
                                 ' otherwise.')
        else:
            pos = xmax - xmin
            pos *= np.random.random_sample([self.npart, self.ndim])
        self._pos = pos.copy()

    def _init_vel(self, vel0):
        if vel0 is None:
            vel0 = 0.0
        if np.isscalar(vel0):
            vel = float(vel0)
            vel *= np.random.standard_normal([self.npart, self.ndim])
        # Sanity checks.
        else:
            vel = np.asarray(vel0, dtype=float)
            if vel.shape != (self.npart, self.ndim):
                raise ValueError('vel0 must have shape (npart, ndim)')
        self._vel = vel.copy()

    def _compute_density(self):
        # Get mesh points for each ptcl. Take modulo to assign xi=L to xi=0,
        # such that mesh points can be integers in range [0, L-1].
        self._meshpts = np.rint(self.pos).astype('int') % self.ngrid
        # for quick access in grid-like arrays
        self._meshinds = tuple(self._meshpts[:, i] for i in range(self.ndim))

        # For each particle, assign density (mass bc unitary cells) to mesh
        # point.
        # NGP assigment scheme
        # Use ndim histogram to save some time (no loop)
        edges = np.linspace(0, self.ngrid-1, num=self.ngrid+1)
        edges = np.repeat([edges], self.ndim, axis=0)
        hist = np.histogramdd(self._meshpts, bins=edges,
                              weights=self.m.flatten())
        self._density = hist[0]

    def _init_green(self):
        # Compute Green's function in one corner of mesh grid.
        greensize = self.ngrid // 2
        meshrange = np.arange(greensize, dtype=float)
        meshrange = np.repeat([meshrange], self.ndim, axis=0)
        mesh = np.array(np.meshgrid(*meshrange))
        rsqr = np.sum(mesh**2, axis=0)
        rsqr[rsqr < self.soft**2] = self.soft**2
        rsqr += self.soft**2
        r = np.sqrt(rsqr)
        green = 1 / (4*np.pi*r)

        # flip along each dimension
        for i in range(self.ndim):
            green = np.append(green, np.flip(green, axis=i), axis=i)

        self._green = green

    def get_pot(self):
        """Compute model potential
        Get potential for each mesh cell by convoluting pre-calculated Green's
        function. Convolution is performed with 2 fourier transform.
        Returns:
            pot (array): potential at every mesh point.
        """
        # Perform convolution
        pot = np.fft.rfftn(self._green) * np.fft.rfftn(self.density)
        pot = np.fft.irfftn(pot)

        # Because of our periodic Green's function definition used in the FFT,
        # potential is not symmetric wrt ptcl positions, so we need to shift
        # and average to center it back on ptcls.
        for i in range(self.ndim):
            pot = 0.5 * (np.roll(pot, 1, axis=i) + pot)

        # enforce BCs
        if self.bc == 'grounded':
            pad = tuple((slice(1, -1),)) * self.ndim
            pot = np.pad(pot[pad], 1)

        return pot

    def get_fmesh(self):
        """Compute mesh forces
        Get force components for each mesh point by differentiating current
        potential grid.
        Returns:
            fmesh (array): forces at each grid point. Array of shape
            (ndim, (ngrid)*ndim) where first axis 0 is the x and y components.
        """
        # Init force array.
        shape = [self.ndim]
        shape.extend([self.ngrid]*self.ndim)
        fmesh = np.zeros(shape, dtype=float)

        # Get pot.
        pot = self.get_pot()

        # periodic gradient in each direction
        for i in range(self.ndim):
            fmesh[i] = 0.5*(np.roll(pot, 1, axis=i) - np.roll(pot, -1, axis=i))

        # Times density and -1 to get actual forces
        fmesh *= - self.density * self.G

        return fmesh

    def get_fpart(self):
        """Compute particle forces
        Get force components for each particle by interpolating from mesh
        forces.
        Returns:
            fpart (array): forces on particle along each axis. Array has shape
                           (npart, 2).
        """
        # Get mesh forces and transpose such that callig fmesh[i,j] will give
        # x and y components for mesh cell (i,j).
        fmesh = np.moveaxis(self.get_fmesh(), 0, -1)

        # Interpolate for each praticle using same scheme as in mass
        # assignment.
        # NGP interpolation.
        fpart = fmesh[self._meshinds]

        return fpart

    def get_energy(self):
        """Get energy of the system
        Kinetic energy is calculated using ptcl velocities and masses.
        Potential energy is obtained with get_pot(), summing over all cells
        """
        # compute kin and pot as in class
        kin = np.sum(self.m * self.vel**2)
        pot = -0.5*np.sum(np.sum(self.get_pot()) * self.density) * self.G

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

            # update velocity
            self._vel += forces * self.dt / self._m

            # update position
            self._pos += self.vel * self.dt
            self._pos = self._pos % self.ngrid

            # update grid
            self._compute_density()

        return self.get_energy()
