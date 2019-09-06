"""Practicing Python"""

#Q00: Import modules here
import numpy as np
import matplotlib.pyplot as plt
###YOUR CODE HERE!!!

#Q01: Reviewing data types

#YOU SHOULD NOT USE FUNCTIONS THAT ALREADY EXIST TO REWRITE THESE FUNCTIONS
#(HINT - except for this problem)
def make_number(s):
    """
    This function takes in a string and should return a floating point number.

    >>> make_number('8.7')
    8.7
    >>> make_number('22.8')
    22.8
    >>> make_number('19')
    19.0
    """
    ###YOUR CODE HERE!!!

#Q02: Reviewing conditionals and for loops

#YOU SHOULD NOT USE FUNCTIONS THAT ALREADY EXIST TO REWRITE THOSE FUNCTIONS
def true_false(a):
    """
    This function should return true if the array inputted is all positive and false if 
    the array contains a negative number. 

    >>> true_false([1,2,3])
    True
    >>> true_false([5,2,-100])
    False
    >>> true_false([1,3,6,100,4000,577])
    True
    """
    ###YOUR CODE HERE!!!



#Q02: Reviewing dictionaries

def dict_print(d):
    """
    For each key-value pair in the input dictionary, print the key and value in this format:
    "The key is fruit and the value is pears"

    >>> dict_print({"fruit":"apple", "vegetable":"carrot"})
    The key is fruit and the value is apple.
    The key is vegetable and the value is carrot.
    """
    ###YOUR CODE HERE!!!


#Q03a: Plotting

def plot_1Dfunction(func, start, end, n):
    """
    func can be called like this func(1) = 10

    Plot the function of one variable between start and end for n values. 

    Hint: Try to use numpy's linspace

    >>> plot_1Dfunction(np.sin, 0, 2*np.pi, 50)
    """
    ###YOUR CODE HERE!!!

#Q03b: Plotting

def plot_2Dfunction(func, start1, end1, n1, start2, end2, n2):
    """
    func can be called like this func(1, 2) = 300

    Plot the function of two variables inputted for n1 values of the first variable
    and n2 values of the second value. 

    Hint: Try to use numpy's meshgrid

    >>> plot_2Dfunction(lambda x,y : x*y, 0, 1, 50, 0, -1, 50)
    """
    ###YOUR CODE HERE!!!



if __name__ == "__main__":
    import doctest
    doctest.testmod()
