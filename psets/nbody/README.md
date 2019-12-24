# Final N-Body Project

## Commented Results and Analysis
**[A notebook](https://github.com/vandalt/phys512/tree/master/psets/nbody/nbody.ipynb) contains results and discussion for all parts, as well as an overview of the code.**
The notebook may take some time to load, but all what is presented there is also saved in .gif and .txt files. Each part also has a corresponding ".py" file in which the problem is set up for a specific part. All results are shown as animations and placed in the ["gifs" folder](https://github.com/vandalt/phys512/tree/master/psets/nbody/gifs). The energy data for each run is in a text file, in the ["energy" folder](https://github.com/vandalt/phys512/tree/master/psets/nbody/energy). Both animatinos and energy data are also presented in the notebook.

## Modules
The NBody object and other utility functions are defined in separate files:
    
- [nbody.py](https://github.com/vandalt/phys512/tree/master/psets/nbody/nbody.py): contains class defintion of NBody object. Main file for this project.
- [utils.py](https://github.com/vandalt/phys512/tree/master/psets/nbody/utils.py): contains utility functions to plot the model and save energy information.

More information is available in the comments and in the notebook.

I used python 3.7.3, but I tried to make compatible with python 2.7 (there may still be errors in python 2.7, I haven't tried).
