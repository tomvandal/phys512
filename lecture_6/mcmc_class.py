import numpy as np
from matplotlib import pyplot as plt

def fun(params,x):
    #params[0]=offset
    #params[1]=amplitude
    #params[2]=period
    #params[3]=phase
    return params[0]+params[1]/np.cos(x*2*np.pi/params[2]+params[3])

def take_step(scale_factor=1.0):
    param_scalings=np.asarray([0.1,0.1,0.1,0.01])
    return scale_factor*param_scalings*np.random.randn(len(param_scalings))
def take_step_cov(covmat):
    mychol=np.linalg.cholesky(covmat)
    return np.dot(mychol,np.random.randn(covmat.shape[0]))



x=np.linspace(-1,1,1000)
params_true=np.asarray([0.5,2,9.0,0.5])
y_true=fun(params_true,x)
noise=np.ones(len(x))*0.3
y=y_true+np.random.randn(len(x))*noise

params=params_true+take_step()
nstep=50000
chains=np.zeros([nstep,len(params)]) #keep track of where the chain went
chisqvec=np.zeros(nstep)
chisq=np.sum( (y-fun(params,x))**2/noise**2)
scale_fac=0.5
for i in range(nstep):
    new_params=params+take_step(scale_fac)
    new_model=fun(new_params,x)
    new_chisq=np.sum( (y-new_model)**2/noise**2)
    
    delta_chisq=new_chisq-chisq
    prob=np.exp(-0.5*delta_chisq)
    accept=np.random.rand(1)<prob
    if accept:
        params=new_params
        model=new_model
        chisq=new_chisq
    chains[i,:]=params
    chisqvec[i]=chisq

mycov=np.cov(chains[nstep/2:,:].T)
npar=len(params_true)
chains_new=np.zeros([nstep,npar])
params=np.mean(chains,axis=0)
chisq=np.sum( (y-fun(params,x))**2/noise**2)
scale_fac=0.5
chisqvec_new=np.zeros(nstep)
for i in range(nstep):
    new_params=params+take_step_cov(mycov)*scale_fac
    new_model=fun(new_params,x)
    new_chisq=np.sum( (y-new_model)**2/noise**2)
    
    delta_chisq=new_chisq-chisq
    prob=np.exp(-0.5*delta_chisq)
    accept=np.random.rand(1)<prob
    if accept:
        params=new_params
        model=new_model
        chisq=new_chisq
    chains_new[i,:]=params
    chisqvec_new[i]=chisq
    
fit_params=np.mean(chains_new,axis=0)




plt.ion()

