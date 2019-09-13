import numpy

#make an array of ones
x=numpy.ones([5,5])
print type(x)
#x*x does an element-by-element multiply for arrays
xx=x*x
print xx
#but we can get a matrix-type product with numpy.dot
x2=numpy.dot(x,x)
print x2



#alternatively, we can recast our array as a matrix
y=numpy.matrix(x)
print type(y)
#now multiply does a matrix multiply
yy=y*y
print yy
#but we can still do an element-wise multiply
y2=numpy.multiply(y,y)
print y2
