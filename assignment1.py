"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test
        # result = lambda x:x;
        #
        # return result
        x_values = np.linspace(a, b, n)

        xy_values = np.array([[x, f(x)] for x in x_values], dtype=object)

        def get_bezier_coef(points):
            # since the formulas work given that we have n+1 points
            # then n must be this:
            n = len(points) - 1

            # build coefficents matrix
            C = 4 * np.identity(n, dtype=object)
            np.fill_diagonal(C[1:], 1)
            np.fill_diagonal(C[:, 1:], 1)
            C[0, 0] = 2
            C[n - 1, n - 1] = 7
            C[n - 1, n - 2] = 2

            # build points vector
            P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
            P[0] = points[0] + 2 * points[1]
            P[n - 1] = 8 * points[n - 1] + points[n]

            a = np.diag(C, -1)
            b = np.diag(C, 0)
            c = np.diag(C, 1)

            def TDMAsolver(a, b, c, d):
                nf = len(d)  # number of equations
                ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
                for it in range(1, nf):
                    mc = ac[it - 1] / bc[it - 1]
                    # mc = ac[it] / bc[it - 1]
                    bc[it] = bc[it] - mc * cc[it - 1]
                    dc[it] = dc[it] - mc * dc[it - 1]

                xc = bc
                xc[-1] = dc[-1] / bc[-1]

                for il in range(nf - 2, -1, -1):
                    xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

                return xc

            Ax = TDMAsolver(a, b, c, [px[0] for px in P])
            Ay = TDMAsolver(a, b, c, [py[1] for py in P])

            A = np.hstack((Ax.reshape(-1, 1), Ay.reshape(-1, 1)))

            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + points[n]) / 2

            return A, B

        # returns the general Bezier cubic formula given 4 control points
        def get_cubic(a, b, c, d):
            return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,
                                                                                                              2) * c + np.power(
                t, 3) * d

        # return one cubic curve for each consecutive points
        def get_bezier_cubic(points, A, B):
            dict_func = {}
            for i in range(len(points) - 1):
                key = (points[i][0], points[i][1])
                value = get_cubic(points[i], A[i], B[i], points[i + 1])
                dict_func[key] = value
            return dict_func

        A, B = get_bezier_coef(xy_values)
        dict_functions = get_bezier_cubic(xy_values, A, B)

        def g(x):
            lst_points = list(dict_functions.keys())
            res = None
            for i in range(len(lst_points) - 1):
                if lst_points[i][0] <= x <= lst_points[i + 1][0]:  # Checking if the x is in the range
                    t = (x - lst_points[i][0]) / (lst_points[i + 1][0] - lst_points[i][0])
                    func = dict_functions.get(lst_points[i])(t)
                    res = func[1]

            if res is None:
                t = (x - lst_points[len(lst_points) - 1][0]) / (
                        x_values[-1] - lst_points[len(lst_points) - 1][0])  # Normalize the x value
                func = dict_functions.get(lst_points[len(lst_points) - 1])(t)
                res = func[1]
            return res

        return g


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
