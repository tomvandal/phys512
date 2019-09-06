import numpy as np
from matplotlib import pyplot as plt

x=np.linspace(-1,1,5)
y=np.sqrt(np.abs(x))
y=np.abs(x)

xx=np.linspace(-1.2,1.2,1000)

plt.clf()
plt.plot(x,y,'*')
for i in range(len(x)-2):
    pp=np.polyfit(x[i:i+3],y[i:i+3],2)
    yy=np.polyval(pp,xx)
    plt.plot(xx,yy)
plt.savefig('parabolas.png')

