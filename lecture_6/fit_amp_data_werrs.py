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


x=nu[ii]
y=g1_cold[ii]
y2=g2_cold[ii]

coeffs=np.polyfit(x,y,6)
pred=np.polyval(coeffs,x)
plt.clf()
plt.plot(x,y)
plt.plot(x,pred)

noise=np.sqrt(np.mean( (pred-y)**2))

ord=5
n=len(x)

xx=2*(x-x[0])/(x[-1]-x[0])-1
print('range is ' + repr([xx.min(),xx.max()]))

A=np.zeros([n,ord+1])
A[:,0]=1.0
A[:,1]=xx
for i in range(1,ord):
    A[:,i+1]=2*xx*A[:,i]-A[:,i-1]

Ninv=np.eye(n)/noise**2

lhs=np.dot(A.transpose(),np.dot(Ninv,A))
rhs=np.dot(A.transpose(),np.dot(Ninv,y))
rhs2=np.dot(A.transpose(),np.dot(Ninv,y2))
lhs_inv=np.linalg.inv(lhs)
fitp=np.dot(lhs_inv,rhs)

fit_err_mat=np.dot(A,np.dot(lhs_inv,A.transpose()))
fit_err_vec=np.sqrt(np.diag(fit_err_mat))
pred=np.dot(A,fitp)

fitp2=np.dot(lhs_inv,rhs2)
pred2=np.dot(A,fitp2)



plt.ion()
plt.clf()
#plt.plot(x,np.abs(pred2-pred))
#plt.plot(x,fit_err_vec)
plt.plot(x,y)
plt.plot(x,pred)
plt.plot(x,pred+fit_err_vec)
plt.plot(x,pred-fit_err_vec)
plt.savefig('error_band.png')
