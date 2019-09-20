import numpy as np
from matplotlib import pyplot as plt

import xlrd

def read_gain(fname):
    #read an excel file into python
    crud=xlrd.open_workbook(fname)
    sheet=crud.sheet_by_index(0)

    nu=np.asarray(sheet.col_values(0))
    nu=nu/1e6 #convert frequency from Hz to MHz
    gain=np.asarray(sheet.col_values(1))
    
    return nu,gain


nu,g1_cold=read_gain('A02_GAIN_MIN20_293.xlsx')
nu,g2_cold=read_gain('A02_GAIN_MIN20_293_2.xlsx')

nu,g1_hot=read_gain('A02_GAIN_MIN20_373.xlsx')
nu,g2_hot=read_gain('A02_GAIN_MIN20_373_2.xlsx')


nu_min=42
ii=nu>nu_min #True for nu>nu_min, False otherwise
#plt.ion()
plt.clf()
plt.plot(nu[ii],g1_cold[ii])
plt.plot(nu[ii],g2_cold[ii])
plt.plot(nu[ii],g1_hot[ii])
plt.plot(nu[ii],g2_hot[ii])
plt.legend(['Run 1 cold','Run 2 cold','Run 1 hot','Run 2 hot'])
plt.xlabel('Frequency (MHz)')
plt.ylabel('Gain')
plt.savefig('all_gains.png')



x=nu[ii]
y=g1_hot[ii]
coeffs=np.polyfit(x,y,6)
pred=np.polyval(coeffs,x)
plt.clf()
plt.plot(x,y)
plt.plot(x,pred)

noise=np.sqrt(np.mean( (pred-y)**2))


for ord in range(3,8):
    coeffs=np.polyfit(x,y,ord)
    pred=np.polyval(coeffs,x)
    chisq=np.sum( (pred-y)**2/noise**2)
    print('chisquared for ' + repr(ord) + ' order is ' + repr(chisq))

print('residual squared error is ' + repr(np.mean((pred-y)**2)))
