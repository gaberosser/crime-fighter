__author__ = 'gabriel'
import numpy as np
from scipy.stats import multivariate_normal
import c1
import c2
PI = np.pi


class MultivariateNormal():

    def __init__(self, mean, vars):
        self.mean = np.array(mean, dtype=float)
        self.vars = np.array(vars, dtype=float)
        self.ndim = self.vars.size

    def pdf(self, x):
        """ Input is an ndarray of dims N x ndim. """
        try:
            shp = x.shape
        except AttributeError:
            shp = np.array(x, dtype=np.float64).shape

        ax = 1

        if not shp and self.ndim == 1:
            # OK - passed a float, 1D implementation
            return self.norm1d(x, self.mean, self.vars)
        elif len(shp) == 1 and self.ndim == 1:
            # OK - passed a 1D array, 1D implementation
            return self.norm1d(x, self.mean, self.vars)
        elif len(shp) == 1:
            ax = 0
        elif shp[-1] != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")

        a = np.power(2*PI, -self.ndim * 0.5)
        b = np.prod(np.power(self.vars, -0.5))
        c = np.exp(-np.sum((x - self.mean)**2 / (2 * self.vars), axis=ax))
        return a * b * c

    @staticmethod
    def norm1d(x, mu, var):
        return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))

    def normnd(self, mu, var, *args):
        # each input is a (1 x self.ndim) array
        a = np.power(2 * PI, self.ndim/2.)
        b = np.prod(np.sqrt(var))
        c = -np.sum((args - mu)**2 / (2 * var))
        return np.exp(c) / (a * b)


class MultivariateNormalScipy():
    """
        This method was made for comparison with the simpler diagonal covariance method.
        It is substantially slower!  Don't use unless a covariance dependency is required.
    """

    def __init__(self, mean, vars):
        self.ndim = len(vars)
        self.mean = mean
        self.vars = vars
        # create covariance matrix
        self.cov = np.zeros((self.ndim, self.ndim))
        self.cov[np.diag_indices_from(self.cov)] = self.vars

    def pdf(self, *args):
        """ Each input is an ndarray of same dims, representing a value of one dimension.
            Result is broadcast of these arrays, hence same shape. """
        if len(args) != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")

        shapes = [np.array(x).shape for x in args]
        for i in range(self.ndim - 1):
            if shapes[i+1] != shapes[i]:
                raise AttributeError("All input arrays must have the same shape")

        it = np.nditer(args + (None,))
        for x in it:
            x[self.ndim][...] = multivariate_normal.pdf(it[:-1], mean=self.mean, cov=self.cov)

        return it.operands[self.ndim]