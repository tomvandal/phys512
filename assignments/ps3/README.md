# Solutions for problem set 3.

## Solutions for each problem
**[The notebook](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/ps3.ipynb) contains my solutions for all the problems**
Note: [Q1](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/q1.py) and [Q2](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/q2.py) have their own script (which are copied in the notebook), but for question 3 and 4 I used [run_mcmc.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/run_mcmc.py) to run the MCMC and [importance.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/importance.py) to do importance sampling. Only the chains analysis part of run_mcmc was copied in the notebook (and the chains are loaded from text files).

## Modules
The files that are not mentionned above are simple modules used throughout the problem set:
    
- [cmb_methods.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/cmb_methdos.py): contains CAMB related stuff.
- [opt_methods.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/opt_methods.py): contains all optimization related methods (Grad, LMA, Cov, MCMC, Poposal distributions).
- [plots.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps3/plots.py): contains plotting functions for MCMC and for CMB.

I used python 3.7.3, but I tried to make compatible with python 2.7 (there may still be errors in python 2.7, I haven't tried).
