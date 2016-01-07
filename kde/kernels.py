__author__ = 'gabriel'

import numpy as np
import math
from scipy import special
from data.models import DataArray, exp, NetworkData, NetworkSpaceTimeData, CartesianSpaceTimeData, CartesianData
from network import utils

PI = np.pi
root2 = np.sqrt(2)
rootpi = np.sqrt(PI)

# helper functions

def normcdf(x, mu, std):
    return 0.5 * (1 + special.erf((x - mu) / (np.sqrt(2) * std)))


class BaseKernel(object):
    """
    Abstract class to ensure basic conformation of all derived kernels
    """
    data_class = DataArray

    @property
    def ndim(self):
        raise NotImplementedError()

    def normalisation_constants(self):
        raise NotImplementedError()

    def prep_input(self, x, expctd_dims=None):
        if not isinstance(x, self.data_class):
            x = self.data_class(x, copy=False)
        if expctd_dims and x.nd != expctd_dims:
            raise AttributeError("Incorrect dimensions for input variable")
        return x

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

    def __init__(self, mean, stdevs):
        self.mean = np.array(mean, dtype=float)
        self.stdevs = np.array(stdevs, dtype=float)
        self.int_constants = self.normalisation_constants()

    @property
    def vars(self):
        return self.stdevs ** 2

    @property
    def ndim(self):
        return self.stdevs.size

    def normalisation_constants(self):
        return root2 * rootpi * self.stdevs

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
        return normcdf(x.toarray(0), self.mean[dim], self.stdevs[dim])


class RadialTemporal(MultivariateNormal):

    def __init__(self, mean, stdevs):
        """
        Kernel with temporal and radial spatial components (t, r), Gaussian in both.
        The main difference between this and MultivariateNormal is in the NORMALISATION: the expressions for
        the integral of a kernel about (r, theta) are complicated.
        NB: this is only valid for 2D space.  The integration constants will differ for the spherical system.
        :param mean:
        :param vars:
        :return:
        """
        if len(mean) != 2 or len(stdevs) != 2:
            raise AttributeError("Length of means and vars array must be 2")
        super(RadialTemporal, self).__init__(mean, stdevs)
        # set up norming here
        self.int_constants = self.normalisation_constants()

    def normalisation_constants(self):
        # time component
        i_tot_t = root2 * rootpi * self.stdevs[0]

        # radial component
        # NB this is only valid for 2D space
        m = self.mean[1]
        s = self.stdevs[1]
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
        s = self.stdevs[dim]
        if dim == 0:
            return normcdf(data, m, s)
        elif dim == 1:
            cdf = 2 * PI * s ** 2 * (np.exp(-m ** 2 / 2 / s ** 2) - np.exp(-(data - m) ** 2 / 2 / s ** 2))
            cdf += m * s * root2 * rootpi * PI * (special.erf((data - m) / s / root2) + special.erf(m / s / root2))
            # solution only valid for r >= 0, plus very small floating point error at r == 0
            if data.size > 1:
                cdf[data <= 0] = 0.
            else:
                cdf = 0. if data <= 0 else cdf
            return cdf / self.int_constants[1]


class SpaceNormalTimeGteZero(MultivariateNormal):
    """
    Same as multivariate normal KDE, but time component is truncated at zero and renormalised.
    """
    def normalisation_constants(self):
        a = root2 * rootpi * self.stdevs
        a[0] *= 1 - normcdf(0, self.mean[0], self.stdevs[0])
        return a

    def pdf(self, x, dims=None):
        if dims:
            ndim = len(dims)
        else:
            dims = range(self.ndim)
            ndim = self.ndim

        x = self.prep_input(x, ndim)

        res = super(SpaceNormalTimeGteZero, self).pdf(x, dims=dims)
        if 0 in dims:
            res[x.toarray(0) < 0] = 0.
        return res


class SpaceNormalTimeExponential(BaseKernel):

    def __init__(self, mean, stdevs):
        # first element of mean is actually the LOCATION of the exponential distribution
        # first element of vars is actually the MEAN of the exponential distribution
        self.mean = np.array(mean, dtype=float)
        self.stdevs = np.array(stdevs, dtype=float)

        # construct a sub-kernel from MultivariateNormal
        self.mvn = MultivariateNormal(self.mean, self.stdevs)

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
            scale = self.stdevs[0]
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
            scale = self.stdevs[0]
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
            res = special.erf((x - self.mean[0]) / (math.sqrt(2) * self.stdevs[0]))
            res -= special.erf((-x - self.mean[0]) / (math.sqrt(2) * self.stdevs[0]))
            res[x < 0] = 0.
            return 0.5 * res
        else:
            return super(SpaceTimeNormalReflective, self).marginal_cdf(x, dim=dim)


class RadialReflectedTemporal(SpaceTimeNormalReflective, RadialTemporal):
    pass


    # def pdf(self, x, dims=None):
    #     """ Input is an ndarray of dims N x ndim.
    #         This may be a data class or just a plain array.
    #         If dims is specified, it is an array of the dims to include in the calculation """
    #
    #     if dims:
    #         ndim = len(dims)
    #     else:
    #         dims = range(self.ndim)
    #         ndim = self.ndim
    #
    #     x = self.prep_input(x, ndim)
    #     res = super(RadialReflectedTemporal, self).pdf(x, dims=dims)
    #
    #     # test whether temporal dimension was included
    #     if 0 in dims:
    #         new_x = SpaceTimeNormalReflective.time_reversed_array(x)
    #         res += super(RadialReflectedTemporal, self).pdf(new_x, dims=dims)
    #         res[x.toarray(0) < 0] = 0.
    #     return res
    #
    # def marginal_cdf(self, x, dim=0):
    #     if dim == 0:
    #         x = self.prep_input(x, 1).toarray(0)
    #         res = special.erf((x - self.mean[0]) / (math.sqrt(2) * self.stdevs[0]))
    #         res -= special.erf((-x - self.mean[0]) / (math.sqrt(2) * self.stdevs[0]))
    #         res[x < 0] = 0.
    #         return 0.5 * res
    #     else:
    #         return super(RadialReflectedTemporal, self).marginal_cdf(x, dim=dim)


class LinearKernel1D(BaseKernel):

    """
    Simple linear 1D kernel.  Useful for network KDE.
    """

    def __init__(self, h, loc=0., one_sided=False):
        """

        :param h: Bandwidth - kernel decreases linearly from loc to zero over this distance
        :param loc: Central location of kernel, defaults to zero
        :param one_sided: If True then the kernel is only defined for x >= loc
        :return:
        """
        self.h = float(h)
        self.loc = loc
        self.one_sided = one_sided
        self.int_constant = self.normalisation_constants()

    @property
    def ndim(self):
        return 1

    def normalisation_constants(self):
        return self.h ** 2

    def pdf(self, x, **kwargs):
        if self.one_sided:
            t = x - self.loc
            return 2 * (self.h - t) * (t >= 0) * (t <= self.h) / self.int_constant
        else:
            t = np.abs(x - self.loc)
            return (self.h - t) * (t <= self.h) / self.int_constant


class LinearRadialKernel(BaseKernel):
    data_class = CartesianData

    def __init__(self, loc, radius):
        self.loc = np.array(list(loc))
        self.radius = radius
        self.int_constant = self.normalisation_constants()

    @property
    def ndim(self):
        return self.loc.size

    def normalisation_constants(self):
        if self.ndim == 1:
            return self.radius ** 2
        if self.ndim == 2:
            return np.pi / 3. * self.radius ** 3
        if self.ndim == 3:
            return np.pi / 3. * self.radius ** 4
        raise NotImplementedError()

    def pdf(self, x, dims=None):
        if dims:
            raise NotImplementedError()

        x = self.prep_input(x, self.ndim)
        loc = self.data_class(np.tile(self.loc, (x.ndata, 1)))
        d = x.distance(loc)
        res = (self.radius - d) * (d <= self.radius) / self.int_constant

        return res.toarray()


class LinearSpaceExponentialTime(BaseKernel):

    data_class = CartesianSpaceTimeData

    def __init__(self, loc, scale):
        # first element of mean is actually the LOCATION of the exponential distribution
        # first element of vars is actually the MEAN of the exponential distribution
        self.loc = np.array(loc, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.linear_kernel = LinearKernel1D

    @property
    def ndim(self):
        return self.loc.size

    def pdf(self, x, dims=None):

        if dims:
            ndim = len(dims)
        else:
            dims = range(self.ndim)
            ndim = self.ndim

        x = self.prep_input(x, ndim)

        # test whether temporal dimension was included
        if 0 in dims:
            scale = self.scale[0]
            mean = self.loc[0]
            t = x.time.toarray()
            res_t = np.exp((mean - t) / scale) / scale * (t > mean)
            dims.remove(0)
        else:
            res_t = 1

        # if dims are left over, these are the spatial components
        if len(dims):
            s = x.space
            res_s = self.mvn.pdf(s, dims=dims)
        else:
            s = 1

        return t * s

    def marginal_cdf(self, x, dim=0):
        x = self.prep_input(x, 1)
        if dim == 0:
            scale = self.stdevs[0]
            c = 1 - np.exp((self.mean[0] - x) / scale)
            c[c < 0] = 0.
            return c
        else:
            return self.mvn.marginal_cdf(x, dim=dim)

    def partial_marginal_pdf(self, x, dim=0):
        raise NotImplementedError()

class NetworkKernelEqualSplitLinear(BaseKernel):
    """ Okabe equal split spatial network kernel """

    kernel_class = LinearKernel1D

    def __init__(self, loc, bandwidth):
        """
        :param loc: NetPoint instance giving the network location of the source
        :param bandwidth: Float giving the search radius / bandwidth of the network kernel
        NB: no cutoff required as this is already supplied in the NetWalker instance
        """
        self.loc = loc
        self.bandwidth = bandwidth
        self.net_kernel = None
        self.set_net_kernel()
        self.walker = None

    def set_net_kernel(self):
        # two-sided kernel - we are in effect computing an absolute distance, so the integral over all space
        # includes both 'positive and negative distance'
        self.net_kernel = self.kernel_class(self.bandwidth, one_sided=False)

    @property
    def spatial_bandwidth(self):
        return self.bandwidth

    @property
    def ndim(self):
        return 1

    @property
    def n_targets(self):
        return self.walker.n_targets

    @staticmethod
    def prep_input(x):
        if not isinstance(x, NetworkData):
            x = NetworkData(x)
        return x

    def set_walker(self, walker):
        self.walker = walker

    def net_pdf(self, source):
        paths = self.walker.source_to_targets(source, max_distance=self.spatial_bandwidth)
        res = np.zeros(self.n_targets)
        # iterate over all possible paths from this source to targets
        for t_idx, p in paths.iteritems():
            for path in p:
                dist = path.distance_total  # source-target distance
                norm = path.splits_total  # number of splits
                res[t_idx] += self.net_kernel.pdf(dist) / norm
        return res

    def pdf(self, *args, **kwargs):
        return self.net_pdf(self.loc)

    def update_bandwidths(self, bandwidth, **kwargs):
        """ In order to retain the net walker cache, update the bandwidth rather than creating a new instance """
        self.bandwidth = bandwidth
        self.set_net_kernel()


class NetworkTemporalKernelEqualSplit(NetworkKernelEqualSplitLinear):

    """
    Kernel defined on a network, with additional exponential time component.
    This makes use of the Okabe approach to computing a KDE on a network (the 'equal split' algorithm)
    """

    def __init__(self, loc, bandwidths, time_cutoff=None):
        """
        :param loc: [Time (float), NetPoint] means for this kernel
        :param bandwidths: The variance in the time case, the maximum search radius in the network case
        :param time_cutoff: Optional parameter specifying the maximum time difference. Any time differences greater than
        this value automatically generate a density of zero.
        """
        self.time_cutoff = time_cutoff
        super(NetworkTemporalKernelEqualSplit, self).__init__(loc, bandwidths)

    @classmethod
    def compute_cutoffs(cls, bandwidths, tol=1e-4):
        """
        Compute the upper cutoffs coresponding to the supplied tolerance and bandwidths
        This is called by the parent KDE, NOT the kernel instance.
        :param bandwidths:
        :param tol:
        :return: Iterable of the same length as bandwidths
        """
        assert len(bandwidths) == 2, "Wrong number of bandwidths supplied"
        t_cutoff = -bandwidths[0] * np.log(bandwidths[0] * tol)
        assert t_cutoff > 0, "Tolerance is too large: time cutoff is negative"
        d_cutoff = bandwidths[1]
        return [t_cutoff, d_cutoff]

    @property
    def spatial_bandwidth(self):
        return self.bandwidth[1]

    def update_bandwidths(self, bandwidths, time_cutoff=None, **kwargs):
        """ In order to retain the net walker cache, update bandwidths/cutoff rather than creating a new instance """
        assert len(bandwidths) == self.ndim, "Wrong number of bandwidths supplied"
        # self.bandwidth = bandwidths
        super(NetworkTemporalKernelEqualSplit, self).update_bandwidths(bandwidths)
        self.time_cutoff = time_cutoff

    def set_net_kernel(self):
        self.net_kernel = self.kernel_class(self.spatial_bandwidth, one_sided=False)

    @property
    def ndim(self):
        return 2

    def pdf(self, target_times=None, dims=None):
        if dims is not None:
            if any([t not in [0, 1] for t in dims]):
                raise AttributeError("Dims requested are out of range")
            ndim = len(dims)
        else:
            dims = [0, 1]
            ndim = 2

        if 0 in dims:
            # time component
            if target_times is None:
                raise AttributeError("Must supply target times if time component is required")
            t = target_times.toarray() - self.loc[0]
            res_t = np.exp(-t / self.bandwidth[0]) * (t > 0) / self.bandwidth[0]
            if self.time_cutoff is not None:
                res_t *= (t <= self.time_cutoff)
            if ndim == 1:
                return res_t
        if 1 in dims:
            # network/space component
            res_d = super(NetworkTemporalKernelEqualSplit, self).net_pdf(self.loc[1])
            if ndim == 1:
                return res_d

        return res_t * res_d


def illustrate_kernels():
    from matplotlib import pyplot as plt
    means = [1.0, 1.0]
    stdevs = [1.0, 1.0]

    kernels = []

    kernels.append(MultivariateNormal(means, stdevs))
    kernels.append(SpaceTimeNormalOneSided(means, stdevs))
    kernels.append(SpaceTimeNormalReflective(means, stdevs))

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
