"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution
        # X=[0]
        # return X

        g = lambda x: f1(x) - f2(x)
        h = 1e-6
        derivative = lambda x: (g(x + h) - g(x)) / h

        def bisection(g, a, b, tol):
            try:
                if g(a) * g(b) < 0:
                    return None
                while abs(b - a) > tol:
                    c = (a + b) / 2
                    if g(c) == 0:
                        return c
                    elif g(a) * g(c) < 0:
                        b = c
                    else:
                        a = c
                return (a + b) / 2
            except:
                return None

        def newton_raphson(g, derivative, x0, rtol, a, b):
            try:
                counter = 0
                x = x0
                while abs(g(x)) > rtol:
                    counter += 1
                    if counter == 15:
                        return bisection(g, a, b, maxerr)
                    dgx = derivative(x)
                    if dgx != 0:
                        x = x - g(x) / dgx
                    else:
                        return bisection(g, a, b, maxerr)
                return x
            except:
                return None

        list_of_range = np.linspace(a, b + 1, num=100)
        results = []

        for i in range(len(list_of_range) - 1):
            tmp = newton_raphson(g, derivative, list_of_range[i], maxerr, list_of_range[i],
                                 list_of_range[i + 1])

            if tmp is not None:
                if not results and a <= tmp <= b and abs(g(tmp)) <= maxerr:
                    results.append(tmp)
                else:
                    count = 0  # True
                    tmp2 = round(tmp, 0)
                    for result in results:
                        if round(result, 0) == tmp2:  # Prevents duplication
                            count = 1  # False
                            break
                    if count == 0 and a <= tmp <= b and abs(g(tmp)) <= maxerr:
                        results.append(tmp)

        return results


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):
        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        # print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
