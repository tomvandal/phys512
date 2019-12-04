import numpy as np
from mpi4py import MPI 
import time

comm=MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc=comm.Get_size()


#print("I am " + repr(myrank) + " out of " + repr(nproc))

xmin=-5.0
xmax=5.0

tot_len=xmax-xmin
mylen=tot_len/nproc
myxmin=xmin+myrank*mylen
myxmax=xmin+(myrank+1)*mylen
npt=1000000
mynpt=npt//nproc
x=np.linspace(myxmin,myxmax,mynpt+1)
xcent=0.5*(x[0:-1]+x[1:])
dx=mylen/mynpt
#print("process " + repr(myrank) + " is working on " + repr([myxmin,myxmax]))

myintegral=np.sum(dx*np.exp(-0.5*xcent**2))
#print("process " + repr(myrank) + " has dx " + repr(dx) + " with integral " + repr(myintegral))
tot=comm.allreduce(myintegral)
if (myrank==0):
    #time.sleep(0.5)
    print("total integral is " + repr(tot))
comm.barrier()
MPI.Finalize()
#print("analytic prediction is " + repr(2*np.erf(5/np.sqrt(2))))
