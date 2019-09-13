import numpy
from matplotlib import pyplot as plt
plt.ion()
c1=numpy.loadtxt('chain_tt_kiyo_dmpow_decorr_wcov.txt_4')
c2=numpy.loadtxt('chain_tt_kiyo_dmpow_wcov.txt_4')
cft=numpy.fft.rfft(c2[:,-1])
x=numpy.arange(cft.size)
x=x[1:cft.size/2]
cft=cft[1:cft.size/2]

plt.clf()
plt.plot(x,numpy.abs(cft))
ax=plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
