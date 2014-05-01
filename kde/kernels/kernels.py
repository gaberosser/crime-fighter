__author__ = 'gabriel'
import numpy as np
from scipy.stats import multivariate_normal

PI = np.pi


class MultivariateNormal():

    def __init__(self, mean, vars):
        self.mean = np.array(mean, dtype=float)
        self.vars = np.array(vars, dtype=float)
        self.ndim = self.vars.size

    def pdf(self, *args):
        """ Each input is an ndarray of same dims, representing a value of one dimension.
            Result is broadcast of these arrays, hence same shape. """
        if len(args) != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")

        shapes = [np.array(x).shape for x in args]
        for i in range(self.ndim - 1):
            if shapes[i+1] != shapes[i]:
                raise AttributeError("All input arrays must have the same shape")

        it = np.nditer(args + (None,), flags=['buffered'], op_dtypes=['float64'] * (self.ndim + 1), casting='same_kind')
        for x in it:
            x[self.ndim][...] = self.normnd(it[:self.ndim], self.mean, self.vars)
            # x[self.ndim][...] = np.prod([self.norm1d(it[i], self.mean[i], self.vars[i]) for i in range(self.ndim)])

        return it.operands[self.ndim] if it.operands[self.ndim].shape else float(it.operands[self.ndim])

    @staticmethod
    def norm1d(x, mu, var):
        return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))

    def normnd(self, x, mu, var):
        # each input is a (1 x self.ndim) array
        a = np.power(2 * PI, self.ndim/2.)
        b = np.prod(np.sqrt(var))
        c = -np.sum((x - mu)**2 / (2 * var))
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