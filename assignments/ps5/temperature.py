"""
Solve heat equation in box with linear temperature
"""
import numpy as np
import matplotlib.pyplot as plt

# nice plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def solve_heat(dt, dx, k, tmax, xmax, const):
    """FTCS method for von Neumann condition
    We have a 2d box, but we only consider one line at the center, which makes
    this a 1D problem.
    Args:
        dt (float): time steps
        dx (float): space steps
        k  (float): convergence coeff
        tmax (float): max time value to solve for
        ymax (float): max x value of the box
        const (float): constant rate defining VN condition
    Returns:
        x: x values along the box
        temp: temperature along the box
    """
    # convergence factor (from k)
    fact = k * dt/dx**2

    assert fact <= 0.5, ('fact=k*dt/dx**2 must be less than 0.5 to reach'
                         ' convergence')

    x, t = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    c, r = x.size, t.size  # temp dims
    temp = np.zeros([r, c])

    for i in range(r-1):
        for j in range(1, c-1):
            # enforce VN BC
            temp[i, 0] = t[i] * c

            # solve PDE
            temp[i+1, j] = (temp[i, j]
                            + fact * (temp[i, j-1]-2*temp[i, j]+temp[i, j+1]))

    return t, x, temp


def plot_heat(tvals, x, temp, sparse=100, keep=False):
    """Plot of heat propagation
    Args:
        tvals: all time values
        x: x values in the box
        temp: temparature at each point in space and time
        sparse (float): resampling for time values, all pts if None
    """
    labsize = 14
    legsize = 14
    titlesize = 16
    for i in range(tvals.size//sparse):
        if not keep:
            plt.plot(x, temp[sparse*i])
            plt.xlabel('x (m)', fontsize=labsize)
            plt.ylabel('T (K)', fontsize=labsize)
            plt.title('Temparture at t = {:.1f}'.format(tvals[i*sparse]),
                      fontsize=titlesize)
            plt.pause(0.01)
            plt.clf()
        else:
            legmsg = 't = {:.1f}'.format(tvals[i*sparse])
            plt.plot(x, temp[sparse*i], label=legmsg)

    if keep:
        plt.xlabel('x (m)', fontsize=labsize)
        plt.ylabel('T (K)', fontsize=labsize)
        plt.title('Temperature along center of the box', fontsize=titlesize)
        plt.legend(fontsize=legsize)
        plt.show()


if __name__ == '__main__':
    # use solver defined above to get temperature vs x along center of the bo
    dt, dx = 5e-4, 5e-4  # time and space steps
    k = 1e-4  # convergence coeff
    xmax = 0.01  # box width
    tmax = 5  # 5 seconds
    const = 100

    # solve pde
    tvals, x, temp = solve_heat(dt, dx, k, tmax, xmax, const)
    plot_heat(tvals, x, temp, sparse=tvals.size//5, keep=True)
