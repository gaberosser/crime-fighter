__author__ = 'gabriel'

import numpy as np
PI = np.pi

# a few helper functions
def normpdf(x, mu, var):
    return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))


def normcdf(x, mu, var):
    from scipy.special import erf
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2 * var))))


class BaseKernel(object):
    """
    Abstract class to ensure basic conformation of all derived kernels
    """

    @property
    def ndim(self):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def marginal_pdf(self, x, dim=0):
        raise NotImplementedError()

    def marginal_cdf(self, x, dim=0):
        raise NotImplementedError()


class MultivariateNormal(BaseKernel):

    def __init__(self, mean, vars):
        self.mean = np.array(mean, dtype=float)
        self.vars = np.array(vars, dtype=float)

    @property
    def ndim(self):
        return self.vars.size

    def pdf(self, x):
        """ Input is an ndarray of dims N x ndim. """
        try:
            shp = x.shape
        except AttributeError:
            shp = np.array(x, dtype=np.float64).shape

        ax = 1
        if not shp and self.ndim == 1:
            # OK - passed a float, 1D implementation
            return normpdf(x, self.mean[0], self.vars[0])
        elif len(shp) == 1 and self.ndim == 1:
            # OK - passed a 1D array, 1D implementation
            return normpdf(x, self.mean[0], self.vars[0])
        elif len(shp) == 1:
            ax = 0
        elif shp[-1] != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")

        a = np.power(2*PI, -self.ndim * 0.5)
        b = np.prod(np.power(self.vars, -0.5))
        c = np.exp(-np.sum((x - self.mean)**2 / (2 * self.vars), axis=ax))
        return a * b * c

    def marginal_pdf(self, x, dim=0):
        """ Return value is 1D marginal pdf with specified dim """
        return normpdf(x, self.mean[dim], self.vars[dim])

    def marginal_cdf(self, x, dim=0):
        """ Return value is 1D marginal cdf with specified dim """
        return normcdf(x, self.mean[dim], self.vars[dim])


# class MultivariateNormalScipy():
#     from scipy.stats import multivariate_normal
#     """
#         This method was made for comparison with the simpler diagonal covariance method.
#         It is substantially slower!  Don't use unless a covariance dependency is required.
#     """
#
#     def __init__(self, mean, vars):
#         self.ndim = len(vars)
#         self.mean = mean
#         self.vars = vars
#         # create covariance matrix
#         self.cov = np.zeros((self.ndim, self.ndim))
#         self.cov[np.diag_indices_from(self.cov)] = self.vars
#
#     def pdf(self, *args):
#         """ Each input is an ndarray of same dims, representing a value of one dimension.
#             Result is broadcast of these arrays, hence same shape. """
#         if len(args) != self.ndim:
#             raise AttributeError("Incorrect dimensions for input variable")
#
#         shapes = [np.array(x).shape for x in args]
#         for i in range(self.ndim - 1):
#             if shapes[i+1] != shapes[i]:
#                 raise AttributeError("All input arrays must have the same shape")
#
#         it = np.nditer(args + (None,))
#         for x in it:
#             x[self.ndim][...] = self.multivariate_normal.pdf(it[:-1], mean=self.mean, cov=self.cov)
#
#         return it.operands[self.ndim]

class LinearKernel(BaseKernel):

    """
    Simple linear 1D kernel.  Useful for network KDE.
    """

    def __init__(self, h):
        self.h = float(h)

    @property
    def ndim(self):
        return 1

    def pdf(self, x):
        if x <= self.h:
            return (self.h - x) / self.h ** 2
        else:
            return 0.
