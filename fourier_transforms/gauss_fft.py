import numpy
from matplotlib import pyplot
def gauss(x0,sig):
    x=numpy.arange(-10,10,0.1)
    y=numpy.exp(-0.5*(x-x0)**2/sig**2)
    return y

def shift(fk,dx):
    N=len(fk)
    k=numpy.arange(N)
    I=numpy.complex(0,1)
    phase=numpy.exp(-2*numpy.pi*I*k*dx/N)
    newf=fk*phase
    return newf
    

if __name__=='__main__':
    yy=gauss(0,1)
    #pyplot.plot(yy)
    #pyplot.show()
    yft=numpy.fft.fft(yy)
    yft2=shift(yft,-10)
    
    z=numpy.fft.ifft(yft)
    z2=numpy.fft.ifft(yft2)
    
    pyplot.plot(numpy.real(z))
    pyplot.plot(numpy.real(z2))
    pyplot.show()

    #pyplot.plot(numpy.abs(yft))
    #pyplot.savefig('gauss_test.png')
    #pyplot.show()

