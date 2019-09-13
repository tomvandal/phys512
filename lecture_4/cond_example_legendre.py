import numpy
def get_legendre_mat(t,npoly):
    #key relation:  (n+1)P_(n+1)=(2n+1)tP_n - nP_(n-1)
    mat=numpy.zeros([t.size,npoly])
    mat[:,0]=1.0
    if npoly>1:
        mat[:,1]=t
    for i in range(1,npoly-1):
        mat[:,i+1]=((2.0*i+1)*t*mat[:,i]-i*mat[:,i-1])/(i+1.0)
    mat=numpy.matrix(mat)
    return mat

if __name__=='__main__':
    dt=0.001
    t=numpy.arange(-5+dt/2.0,5,dt)
    for npoly in numpy.arange(5,100,5):
        mat=get_legendre_mat(t/5,npoly)
        mm=mat.transpose()*mat
        mm=mm+mm.transpose() #bonus symmetrization
        e,v=numpy.linalg.eig(mm)
        eabs=numpy.abs(e)
        cond=eabs.max()/eabs.min()
        print repr(npoly) + ' Legendre matrix has condition number ' + repr(cond)
