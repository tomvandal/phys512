import numpy
import time
import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.animation as animation

from matplotlib  import pyplot as plt

class particles:
    def __init__(self,m=1.0,npart=1000,soft=0.01,G=1.0,dt=0.1):
        self.opts={}
        self.opts['soft']=soft
        self.opts['n']=npart
        self.opts['G']=G
        self.opts['dt']=dt

        self.x=numpy.random.randn(self.opts['n'])
        self.y=numpy.random.randn(self.opts['n'])
        self.m=numpy.ones(self.opts['n'])*m
        self.vx=0*self.x
        self.vy=self.vx.copy()
    def get_forces(self):
        self.fx=numpy.zeros(self.opts['n'])
        self.fy=0*self.fx
        pot=0
        for i in range(0,self.opts['n']):
            dx=self.x[i]-self.x
            dy=self.y[i]-self.y
            rsqr=dx**2+dy**2
            soft=self.opts['soft']**2
            rsqr[rsqr<soft]=soft
            rsqr=rsqr+self.opts['soft']**2
            r=numpy.sqrt(rsqr)
            r3=1.0/(r*rsqr)
            self.fx[i]=-numpy.sum(self.m*dx*r3)*self.opts['G']
            self.fy[i]=-numpy.sum(self.m*dy*r3)*self.opts['G']
            pot+=self.opts['G']*numpy.sum(self.m/r)*self.m[i]
        return -0.5*pot
    def evolve(self):
        self.x+=self.vx*self.opts['dt']
        self.y+=self.vy*self.opts['dt']
        pot=self.get_forces()
        self.vx+=self.fx*self.opts['dt']
        self.vy+=self.fy*self.opts['dt']
        kinetic=0.5*numpy.sum(self.m*(self.vx**2+self.vy**2))
        return pot+kinetic


if __name__=='__main__':
    plt.ion()
    n=3000
    oversamp=5
    part=particles(m=1.0/n,npart=n,dt=0.1/oversamp)
    plt.plot(part.x,part.y,'*')
    plt.show()
    


    #fig = plt.figure()
    #ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
    #line, = ax.plot([], [], '*', lw=2)

    for i in range(0,10000):
        for ii in range(oversamp):
            energy=part.evolve()
        print energy
        plt.clf()
        plt.plot(part.x,part.y,'*')
        plt.pause(1e-3)
        #line.set_data(part.x,part.y)
        ##plt.show()
        
    #ani = animation.FuncAnimation(fig, animate_points, numpy.arange(30),
    #                          interval=25, blit=False)
    #plt.show()

        
