import numpy as np
from matplotlib import pyplot as plt

fun=np.exp
xmin=-1
xmax=1
x=np.linspace(xmin,xmax,11)
dx=np.median(np.diff(x))
y=fun(x)

xx=np.linspace(x[1],x[-2]-1e-13,1001) #skip the first/last region 
                                #since we aren't double-bracketed

yy_true=fun(xx)
yy=0*yy_true
for i,myx in enumerate(xx):
    j=np.int((myx-xmin)/dx)
    pp=np.polyfit(x[j-1:j+3],y[j-1:j+3],3)
    yy[i]=np.polyval(pp,myx)
plt.clf();
plt.plot(xx,yy_true);
plt.plot(xx,yy);
plt.plot(x,y,'*')
plt.savefig('cubic_interp.png')
print('mean error is ' + repr(np.mean(np.abs(yy-yy_true))))
