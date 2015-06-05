__author__ = 'gabriel'

import numpy as np
import math
from scipy import special
from data.models import DataArray, exp, NetworkData, NetworkSpaceTimeData
from network import utils

PI = np.pi
root2 = np.sqrt(2)
rootpi = np.sqrt(PI)

# helper functions

def normcdf(x, mu, var):
    return 0.5 * (1 + special.erf((x - mu) / (np.sqrt(2 * var))))


class BaseKernel(object):
    """
    Abstract class to ensure basic conformation of all derived kernels
    """

    @property
    def ndim(self):
        raise NotImplementedError()

    def normalisation_constants(self):
        raise NotImplementedError()

    def pdf(self, x, dims=None):
        raise NotImplementedError()

    def marginal_pdf(self, x, dim=0):
        """ Return value is 1D marginal pdf with specified dim """
        return self.pdf(x, dims=[dim])

    def marginal_cdf(self, x, dim=0):
        raise NotImplementedError()

    def partial_marginal_pdf(self, x, dim=0):
        """ Return value is 1D partial marginal pdf:
            full pdf integrated over specified dim """
        dims = range(self.ndim)
        dims.remove(dim)
        return self.pdf(x, dims=dims)


class SpoofKernel(BaseKernel):
    def pdf(self, x, dims=None):
        return 1.


class MultivariateNormal(BaseKernel):

    def __init__(self, mean, vars):
        self.mean = np.array(mean, dtype=float)
        self.vars = np.array(vars, dtype=float)
        self.int_constants = self.normalisation_constants()

    @property
    def ndim(self):
        return self.vars.size

    @staticmethod
    def prep_input(x, expctd_dims=None):
        ## TODO: test the effect this has on speed by profiling
        ## if necessary, can ASSUME a DataArray and implement at a higher level
        if not isinstance(x, DataArray):
            x = DataArray(x)
        if expctd_dims and x.nd != expctd_dims:
            raise AttributeError("Incorrect dimensions for input variable")

        return x

    def normalisation_constants(self):
        return root2 * rootpi * np.sqrt(self.vars)

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

        # a = np.power(2*PI, -ndim * 0.5)

        if ndim == 1:
            b = self.int_constants[dims[0]]
            c = exp(-(x - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]]))
        elif ndim == 2:
            b = self.int_constants[dims[0]] * self.int_constants[dims[1]]
            c = exp(
                - (x.getdim(0) - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]])
                - (x.getdim(1) - self.mean[dims[1]])**2 / (2 * self.vars[dims[1]])
            )
        elif ndim == 3:
            b = self.int_constants[dims[0]] * self.int_constants[dims[1]] * self.int_constants[dims[2]]
            c = exp(
                - (x.getdim(0) - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]])
                - (x.getdim(1) - self.mean[dims[1]])**2 / (2 * self.vars[dims[1]])
                - (x.getdim(2) - self.mean[dims[2]])**2 / (2 * self.vars[dims[2]])
            )
        else:
            b = np.prod(self.int_constants[dims])
            c = exp(-((x - self.mean[dims])**2 / (2 * self.vars[dims])).sumdim())

        # return (c * b * a).toarray(0)
        return (c / b).toarray(0)

    def marginal_cdf(self, x, dim=0):
        """ Return value is 1D marginal cdf with specified dim """
        x = self.prep_input(x, 1)
        return normcdf(x.toarray(0), self.mean[dim], self.vars[dim])


class RadialTemporal(MultivariateNormal):

    def __init__(self, mean, vars):
        """
        Kernel with temporal and radial spatial components (t, r), Gaussian in both.
        The main difference between this and MultivariateNormal is in the NORMALISATION: the expressions for
        the integral of a kernel about (r, theta) are complicated.
        NB: this is only valid for 2D space.  The integration constants will differ for the spherical system.
        :param mean:
        :param vars:
        :return:
        """
        if len(mean) != 2 or len(vars) != 2:
            raise AttributeError("Length of means and vars array must be 2")
        super(RadialTemporal, self).__init__(mean, vars)
        # set up norming here
        self.int_constants = self.normalisation_constants()

    def normalisation_constants(self):
        # time component
        i_tot_t = root2 * rootpi * np.sqrt(self.vars[0])

        # radial component
        # NB this is only valid for 2D space
        m = self.mean[1]
        s = np.sqrt(self.vars[1])
        i_tot_r = 2 * PI * s ** 2 * np.exp(-m ** 2 / 2 / s ** 2)
        i_tot_r += m * s * root2 * rootpi * PI * (1 + special.erf(m / s / root2))

        return np.array([i_tot_t, i_tot_r])

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

        if ndim == 1:
            a = 1 / self.int_constants[dims[0]]
            b = exp(-(x - self.mean[dims[0]]) ** 2 / (2 * self.vars[dims[0]]))
        elif ndim == 2:
            a = 1 / self.int_constants[dims[0]] / self.int_constants[dims[1]]
            b = exp(
                - (x.getdim(0) - self.mean[dims[0]])**2 / (2 * self.vars[dims[0]])
                - (x.getdim(1) - self.mean[dims[1]])**2 / (2 * self.vars[dims[1]])
            )
        else:
            raise NotImplementedError("Only support 3D (time + 2D space)")

        return (b * a).toarray(0)

    def marginal_cdf(self, x, dim=0):
        """ Return value is 1D marginal cdf with specified dim """
        x = self.prep_input(x, 1)
        data = x.toarray(0)
        m = self.mean[dim]
        v = self.vars[dim]
        if dim == 0:
            return normcdf(data, m, v)
        elif dim == 1:
            s = np.sqrt(v)
            cdf = 2 * PI * s ** 2 * (np.exp(-m ** 2 / 2 / s ** 2) - np.exp(-(data - m) ** 2 / 2 / s ** 2))
            cdf += m * s * root2 * rootpi * PI * (special.erf((data - m) / s / root2) + special.erf(m / s / root2))
            # solution only valid for r >= 0, plus very small floating point error at r == 0
            if data.size > 1:
                cdf[data <= 0] = 0.
            else:
                cdf = 0. if data <= 0 else cdf
            return cdf / self.int_constants[1]


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
        x = self.prep_input(x, 1)
        res = super(SpaceTimeNormalOneSided, self).marginal_cdf(x, dim=dim)
        if dim == 0:
            t0 = x.toarray(0) < self.mean[0]
            res[t0] = 0.
            res[~t0] = 2 * res[~t0] - 1.
        return res


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
        if dim == 0:
            x = self.prep_input(x, 1).toarray(0)
            res = special.erf((x - self.mean[0]) / (math.sqrt(2 * self.vars[0])))
            res -= special.erf((-x - self.mean[0]) / (math.sqrt(2 * self.vars[0])))
            res[x < 0] = 0.
            return 0.5 * res
        else:
            return super(SpaceTimeNormalReflective, self).marginal_cdf(x, dim=dim)


class LinearKernel(BaseKernel):

    """
    Simple linear 1D kernel.  Useful for network KDE.
    """

    def __init__(self, h, loc=0., one_sided=True):
        """

        :param h: Bandwidth - kernel decreases linearly from loc to zero over this distance
        :param loc: Central location of kernel, defaults to zero
        :param one_sided: If True then the kernel is only defined for x >= loc
        :return:
        """
        self.h = float(h)
        self.loc = loc
        self.one_sided = one_sided

    @property
    def ndim(self):
        return 1

    def pdf(self, x, **kwargs):
        t = x - self.loc
        if self.one_sided:
            return 2 * (self.h - t) / self.h ** 2 * (t >= 0) * (t <= self.h)
        else:
            return (self.h - t) / self.h ** 2 * (t >= -self.h) * (t <= self.h)


class NetworkKernelEqualSplitLinear(BaseKernel):
    """ Okabe equal split spatial network kernel """

    kernel_class = LinearKernel

    def __init__(self, loc, bandwidth):
        """
        :param loc: NetPoint instance giving the network location of the source
        :param bandwidth: Float giving the search radius / bandwidth of the network kernel
        """
        self.loc = loc
        self.bandwidth = bandwidth
        self.graph = loc.graph
        self.kernel = None
        self.set_kernel()

    def set_kernel(self):
        # one-sided kernel = only ever get positive network distances
        self.kernel = self.kernel_class(self.bandwidth, one_sided=True)

    @property
    def ndim(self):
        return 1

    @staticmethod
    def prep_input(x):
        if not isinstance(x, NetworkData):
            x = NetworkData(x)
        return x

    def pdf(self, x, **kwargs):
        """ x is a NetworkData array of NetPoint objects """
        from network import utils
        verbose = kwargs.pop('verbose', False)
        x = self.prep_input(x)
        paths = utils.network_paths_source_targets(self.graph,
                                                   self.loc,
                                                   x,
                                                   max_search_distance=self.bandwidth,
                                                   verbose=verbose)
        res = np.zeros(x.ndata)
        for i in range(x.ndata):
            val = 0.
            if i in paths:
                for t in paths[i]:
                    #  each of these is a unique path from source to a target
                    degrees = self.graph.g.degree(t[0])
                    a = np.prod(np.array(degrees.values()) - 1)
                    if a == 0:
                        # we passed a node of degree 1 - shouldn't happen
                        import ipdb; ipdb.set_trace()
                    # compute kernel value
                    val += self.kernel.pdf(t[1]) / float(a)
            res[i] = val
        return res

class NetworkTemporalKernelEqualSplit(BaseKernel):

    """
    Kernel defined on a network, with additional Gaussian time component.
    This makes use of the Okabe approach to computing a KDE on a network (the 'equal split' algorithm)
    """

    def __init__(self, loc, bandwidths):
        """
        :param loc: [Time (float), NetPoint] means for this kernel
        :param bandwidths: The variance in the time case, the maximum search radius in the network case
        """
        if len(loc) != 2:
            raise AttributeError("Input loc must have 2 elements")
        if len(bandwidths) != 2:
            raise AttributeError("Input bandwidths must have 2 elements")
        self.loc = loc
        self.bandwidths = np.array(bandwidths, dtype=float)
        self.network_kernel = None
        self.set_kernel()

    def set_kernel(self):
        self.network_kernel = NetworkKernelEqualSplitLinear(self.loc[1], self.bandwidths[1])

    @property
    def ndim(self):
        return 2

    def pdf(self, x, dims=None):
        if dims is not None:
            if any([t not in [0, 1] for t in dims]):
                raise AttributeError("Dims requested are out of range")
            ndim = len(dims)
        else:
            ndim = 2

        if ndim == 1:
            if 0 in dims:
                # time component
                t = x.toarray(0) - self.loc[0]
                return np.exp(-t / self.bandwidths[0]) * (t >= 0) / self.bandwidths[0]
            else:
                # network/space component
                return self.network_kernel.pdf(x)
        elif ndim == 2:
            if not isinstance(x, NetworkSpaceTimeData):
                x = NetworkSpaceTimeData(x)
            return self.pdf(x.time, dims=[0]) * self.pdf(x.space, dims=[1])


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
