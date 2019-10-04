import numpy
import time

n=2**16
x=numpy.random.randn(n)
t1=time.time();y=numpy.fft.fft(x);t2=time.time();t_ref=t2-t1
x=numpy.random.randn(n+1) #this is a prime
t1=time.time();y=numpy.fft.fft(x);t2=time.time();t_plus1=t2-t1
x=numpy.random.randn(n+2) #this is has largest factor 331
t1=time.time();y=numpy.fft.fft(x);t2=time.time();t_plus2=t2-t1
x=numpy.random.randn(n+14) #this is has largest factor 23
t1=time.time();y=numpy.fft.fft(x);t2=time.time();t_plus14=t2-t1
print 'Reference time was ',t_ref
print 'Extending by one increased time by a factor of ',t_plus1/t_ref
print 'Extending by two increased time by a factor of ',t_plus2/t_ref
print 'Extending by  14 increased time by a factor of ',t_plus14/t_ref
