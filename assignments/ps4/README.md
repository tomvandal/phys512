# Solutions for problem set 4.

## Solutions for all events
**[The notebook](https://github.com/tomvandal/phys512/tree/master/assignments/ps4/ps4.ipynb) contains my solutions for all the problems**. This includes answers to the questions from the problem set as well as plots and numerical results for all 4 events.

## Modules
To avoid making the notebook too long or confusing, I implemented all methods in these three files, and I simply run them in the notebook:
    
- [gravwaves.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps4/gravwaves.py): contains a function to search LIGO data for a given event, using methods from utils.py
- [read_ligo.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps4/read_ligo.py): modified version of the script provided to read LIGO data.
- [utils.py](https://github.com/tomvandal/phys512/tree/master/assignments/ps4/utils.py): utility functions used to perform specific tasks (about one per section).

I used python 3.7.3, but I tried to make compatible with python 2.7 (there may still be errors in python 2.7, I haven't tried).
