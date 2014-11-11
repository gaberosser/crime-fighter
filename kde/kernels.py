__author__ = 'gabriel'

import numpy as np
import math
from scipy import special
from data.models import DataArray, Data, exp
import ipdb

PI = np.pi

# a few helper functions
def normpdf(x, mu, var):
    return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))


def normcdf(x, mu, var):
    return 0.5 * (1 + special.erf((x - mu) / (np.sqrt(2 * var))))


class BaseKernel(object):
    """
    Abstract class to ensure basic conformation of all derived kernels
    """

    @property
    def ndim(self):
        raise NotImplementedError()

    def pdf(self, x, dims=None):
        raise NotImplementedError()

    def marginal_pdf(self, x, dim=0):
        raise NotImplementedError()

    def marginal_cdf(self, x, dim=0):
        raise NotImplementedError()

    def partial_marginal_pdf(self, x, dim=0):
        raise NotImplementedError()


class SpoofKernel(BaseKernel):
    def pdf(self, x, dims=None):
        return 1.


class MultivariateNormal(BaseKernel):

    def __init__(self, mean, vars):
        self.mean = np.array(mean, dtype=float)
        self.vars = np.array(vars, dtype=float)

    @property
    def ndim(self):
        return self.vars.size

    @staticmethod
    def prep_input(x, expctd_dims=None):
        ## TODO: test the effect this has on speed by profiling
        ## if necessary, can ASSUME a DataArray and implement at a higher level
        if not isinstance(x, Data):
            x = DataArray(x)
        if expctd_dims and x.nd != expctd_dims:
            import ipdb; ipdb.set_trace()
            raise AttributeError("Incorrect dimensions for input variable")

        return x

    def pdf(self, x, dims=None):
        """ Input is an ndarray of dims N x ndim.
            This may be a data class or just a plain array.
            If dims is specified, it is an array of the dims to include in the calculation """

        if dims:
            ndim = len(dims)
        else:
            dims = range(self.ndim)
            ndim = self.ndim

        x = self.prep_input(x, ndim)

        a = np.power(2*PI, -ndim * 0.5)

        if ndim == 1:
            b = 1. / np.sqrt(self.vars[dims[0]])
            c = exp(-(x - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]]))
        elif ndim == 2:
            b = 1. / np.sqrt(self.vars[dims[0]] * self.vars[dims[1]])
            c = exp(
                - (x.getdim(0) - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]])
                - (x.getdim(1) - self.mean[dims[1]])**2 / (2 * self.vars[dims[1]])
            )
        elif ndim == 3:
            b = 1. / np.sqrt(self.vars[dims[0]] * self.vars[dims[1]] * self.vars[dims[2]])
            c = exp(
                - (x.getdim(0) - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]])
                - (x.getdim(1) - self.mean[dims[1]])**2 / (2 * self.vars[dims[1]])
                - (x.getdim(2) - self.mean[dims[2]])**2 / (2 * self.vars[dims[2]])
            )
        else:
            b = np.prod(np.power(self.vars[dims], -0.5))
            c = exp(-((x - self.mean[dims])**2 / (2 * self.vars[dims])).sumdim())

        return (c * b * a).toarray(0)

    def marginal_pdf(self, x, dim=0):
        """ Return value is 1D marginal pdf with specified dim """
        return self.pdf(x, dims=[dim])

    def marginal_cdf(self, x, dim=0):
        """ Return value is 1D marginal cdf with specified dim """
        x = self.prep_input(x, 1)
        ## FIXME: old normcdf function does not support new data model
        return normcdf(x, self.mean[dim], self.vars[dim]).toarray(0)

    def partial_marginal_pdf(self, x, dim=0):
        """ Return value is 1D partial marginal pdf:
            full pdf integrated over specified dim """
        dims = range(self.ndim)
        dims.remove(dim)
        return self.pdf(x, dims=dims)


class SpaceNormalTimeExponential(BaseKernel):

    def __init__(self, mean, vars):
        # first element of mean is actually the LOCATION of the exponential distribution
        # first element of vars is actually the MEAN of the exponential distribution
        self.mean = np.array(mean, dtype=float)
        self.vars = np.array(vars, dtype=float)

        # construct a sub-kernel from MultivariateNormal
        self.mvn = MultivariateNormal(self.mean, self.vars)

    @staticmethod
    def prep_input(x, expctd_dims=None):
        return MultivariateNormal.prep_input(x, expctd_dims=expctd_dims)

    @property
    def ndim(self):
        return self.mean.size

    def pdf(self, x, dims=None):

        if dims:
            ndim = len(dims)
        else:
            dims = range(self.ndim)
            ndim = self.ndim

        x = self.prep_input(x, ndim)

        # test whether temporal dimension was included
        if 0 in dims:
            scale = np.sqrt(self.vars[0])
            t = np.exp((self.mean[0] - x.getdim(0)) / scale) / scale
            t = t.toarray(0)
            t0 = x.toarray(0) < self.mean[0]
            t[t0] = 0.
            dims.remove(0)
        else:
            t = 1

        # if dims are left over, these are the spatial components
        if len(dims):
            # extract required dims from x
            this_x = x.getdim(dims[0])
            for i in dims[1:]:
                this_x = this_x.adddim(x.getdim(i))
            s = self.mvn.pdf(this_x, dims=dims)
        else:
            s = 1

        return t * s

    def marginal_cdf(self, x, dim=0):
        x = self.prep_input(x, 1)
        if dim == 0:
            scale = np.sqrt(self.vars[0])
            c = 1 - np.exp((self.mean[0] - x) / scale)
            c[c < 0] = 0.
            return c
        else:
            return self.mvn.marginal_cdf(x, dim=dim)

    def partial_marginal_pdf(self, x, dim=0):
        raise NotImplementedError()


class SpaceTimeNormalOneSided(MultivariateNormal):
    """
    Variant of the n-dimensional Gaussian kernel, but first dim is ONE-SIDED - useful for space-time purposes.
    The mean here has the usual definition for dims > 1, but for the first dim it gives the EQUIVALENT (i.e. the first
    non-zero point of the distribution).
    The vars have the usual definition for dims > 1 and for the first dim it is the equivalent one-sided variance.
    """
    def pdf(self, x, dims=None):

        if dims:
            ndim = len(dims)
        else:
            dims = range(self.ndim)
            ndim = self.ndim

        x = self.prep_input(x, ndim)
        if dims is None:
            dims = range(self.ndim)
        res = super(SpaceTimeNormalOneSided, self).pdf(x, dims=dims)
        # test whether temporal dimension was included
        if 0 in dims:
            t0 = x.toarray(0) < self.mean[0]
            t1 = x.toarray(0) >= self.mean[0]
            res[t0] = 0.
            res[t1] *= 2
        return res

    def marginal_cdf(self, x, dim=0):
        x = self.prep_input(x)
        if dim == 0:
            res = special.erf((x - self.mean[0]) / (math.sqrt(2 * self.vars[0])))
            t0 = x.toarray(0) < self.mean[0]
            res[t0] = 0.
            return res
        else:
            return super(SpaceTimeNormalOneSided, self).marginal_cdf(x, dim=dim)


class SpaceTimeNormalReflective(MultivariateNormal):
    """
    Variant of the n-dimensional Gaussian kernel, but first dim is REFLECTED at t=0 - useful for space-time purposes.
    The mean here has the usual definition for dims > 1, but for the first dim it gives the EQUIVALENT (i.e. the first
    non-zero point of the distribution).
    The vars have the usual definition for dims > 1 and for the first dim it is the equivalent one-sided variance.
    """
    @staticmethod
    def time_reversed_array(arr):
        arr = arr.copy()
        try:
            arr.time *= -1.0
        except AttributeError:
            arr.data[:, 0] *= -1.0
        return arr

    def pdf(self, x, dims=None):

        if dims:
            ndim = len(dims)
        else:
            dims = range(self.ndim)
            ndim = self.ndim

        x = self.prep_input(x, ndim)
        res = super(SpaceTimeNormalReflective, self).pdf(x, dims=dims)

        # test whether temporal dimension was included
        if dims and 0 in dims:
            new_x = self.time_reversed_array(x)
            res += super(SpaceTimeNormalReflective, self).pdf(new_x, dims=dims)
            res[x.toarray(0) < 0] = 0.
        return res

    def marginal_cdf(self, x, dim=0):
        x = self.prep_input(x)
        if dim == 0:
            res = special.erf((x - self.mean[0]) / (math.sqrt(2 * self.vars[0])))
            res -= special.erf((-x - self.mean[0]) / (math.sqrt(2 * self.vars[0])))
            res[x.toarray(0) < 0] = 0.
            return 0.5 * res
        else:
            return super(SpaceTimeNormalReflective, self).marginal_cdf(x, dim=dim)

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


def illustrate_kernels():
    from matplotlib import pyplot as plt
    from analysis import plotting
    means = [1.0, 1.0]
    vars = [1.0, 1.0]

    kernels = []

    kernels.append(MultivariateNormal(means, vars))
    kernels.append(SpaceTimeNormalOneSided(means, vars))
    kernels.append(SpaceTimeNormalReflective(means, vars))

    x = np.linspace(-5, 5, 500)
    ymax = 0.
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)

    for k, ax in zip(kernels, axs):
        p = k.marginal_pdf(x, dim=0)
        ymax = max(ymax, max(p))
        ax.plot(x, p)
        ax.set_xlabel('Time (t)', fontsize=18)
        ax.set_xlim([-3, 5])

    ymax *= 1.02

    for ax in axs:
        ax.set_ylim([0., ymax])
        ax.plot([0, 0], [0, ymax], '--', color='gray')

    # particular additions
    axs[0].set_ylabel('Density', fontsize=18)
    axs[0].fill_between(x[x < 0], 0, kernels[0].marginal_pdf(x, dim=0)[x < 0], edgecolor='none', facecolor='gray')
    p = kernels[0].marginal_pdf(x, dim=0)
    pn = p[x < 0][::-1]
    xn = -x[x < 0][::-1]
    axs[2].plot(x, p, 'k--')
    axs[2].plot(xn, pn, 'k--')

    plt.tight_layout()
