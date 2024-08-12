"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""

import numpy as np
import time
import random
import timeit


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        # result = lambda x: x
        # y = f(1)
        #
        # return result

        start_time = timeit.timeit()
        b_list = []
        y_list = np.array([f(a) for i in range(50)])
        y_mean = np.mean(y_list)
        b_list.append([y_mean])
        time_to_pross = timeit.timeit() - start_time
        max_iter = abs(maxtime / time_to_pross)  # Calculation of the maximum number of iterations
        list_of_values_x = np.linspace(a, b, d + 1)
        for idx_x in range(1, len(list_of_values_x)):
            x = list_of_values_x[idx_x]
            rg = int(max_iter / (len(list_of_values_x)))
            y_list = np.array([f(x) for i in range(rg)])  # A list of y values that receive as input a noisy point
            y_mean = np.mean(y_list)  # the average of the y values
            b_list.append([y_mean])

        A = [[x ** (d - i) for i in range(d + 1)] for x in list_of_values_x]

        At = np.transpose(A)
        AtA = np.dot(At, A)
        AtA_inv = np.linalg.inv(AtA)
        AtA_inv_At = np.dot(AtA_inv, At)
        result = np.dot(AtA_inv_At, b_list)  # linear solution
        return lambda x: sum(result[i][0] * (x ** (d - i)) for i in range(d + 1))  # return the function


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)

# if __name__ == "__main__":
#     unittest.main()

# def check_stav1(self):
# ass4 = Assignment4()


# def f1(x):
#     return math.log(math.log(x))
#
#
# shape = ass4.fit(f1, 3, 10, 20, maxtime=15)
# print(shape)
#
# def f1(x):
#     return 1 / math.log(x)
#
#
# shape = ass4.fit(f1, 3, 16, 10, maxtime=10)
# print(shape)
# from numpy import empty, float64

# def f1(x):
#     if type(x) is float64:
#         return float64(5)
#     if type(x) is int:
#         return 5
#     if type(x) is float:
#         return 5.0
#     a = empty(len(x))
#     a.fill(5)
#     return a
#     # return math.log(math.log(x))
#
#
# shape = ass4.fit(f1, 2, 5, 20, maxtime=20)
# print(shape)
