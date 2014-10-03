__author__ = 'gabriel'
import numpy as np
import operator
from functools import partial
from kde import kernels
import multiprocessing as mp
from models import marginal_icdf_optimise
from contextlib import closing
import time
import ctypes

class KernelCluster(object):

    def __init__(self, data, bandwidths, ktype=None):
        self.ktype = ktype or kernels.MultivariateNormal
        if data.shape != bandwidths.shape:
            raise AttributeError("Dims of data and bandwidths do not match")
        self.data = data
        self.bandwidths = bandwidths
        self.kernels = self._kernels

    @property
    def ndim(self):
        return self.data.shape[1]

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def _kernels(self):
        return [self.ktype(self.data[i], self.bandwidths[i]**2) for i in range(self.ndata)]

    def additive_operation(self, funcstr, data, **kwargs):
        """ Generic interface to call function named in funcstr on the data, handling normalisation and reshaping """
        # better to use a generator here to reduce memory usage:
        return reduce(operator.add, (getattr(x, funcstr)(data, **kwargs) for x in self.kernels))


def runner_parallel(x, fstr=None, fd=None):
    # print "runner(x, %s, %s)" % (fstr, str(fd))
    return x.additive_operation(fstr, fd)


def runner_debug(x, fd=None):
    print "RUNNER DEBUG"
    time.sleep(3)
    return np.random.rand(fd.shape[0]) ** 2


class FixedBandwidthKde(object):
    def __init__(self, data, *args, **kwargs):
        ## TODO: add self.ktype to allow switching kernel type - default to Gaussian

        self.data = data
        if len(data.shape) == 1:
            self.data = np.array(data).reshape((len(data), 1))

        try:
            self.ncpu = kwargs.pop('ncpu', mp.cpu_count())
        except NotImplementedError:
            self.ncpu = 1

        self.bandwidths = None
        self.kernel_clusters = None
        self.set_bandwidths(*args, **kwargs)

    def set_bandwidths(self, *args, **kwargs):
        try:
            bandwidths = kwargs.pop('bandwidths')
        except KeyError:
            bandwidths = self.range_array / float(self.ndata)

        if len(bandwidths) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = np.tile(bandwidths, (self.ndata, 1))
        self.set_kernels()


    def set_kernels(self):
        n_per_cluster = int(self.ndata / self.ncpu)
        self.kernel_clusters = []
        idx = 0
        for i in range(self.ncpu - 1):
            this_data = self.data[idx:(idx + n_per_cluster)]
            this_bandwidths = self.bandwidths[idx:(idx + n_per_cluster)]
            idx += n_per_cluster
            self.kernel_clusters.append(KernelCluster(this_data, this_bandwidths))
        # add remaining data to final cluster
        this_data = self.data[idx:]
        this_bandwidths = self.bandwidths[idx:]
        self.kernel_clusters.append(KernelCluster(this_data, this_bandwidths))

    @property
    def ndim(self):
        return self.data.shape[1]

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def max_array(self):
        return np.max(self.data, axis=0)

    @property
    def min_array(self):
        return np.min(self.data, axis=0)

    @property
    def range_array(self):
        return self.max_array - self.min_array

    @property
    def raw_std_devs(self):
        return np.std(self.data, axis=0, ddof=1)

    def _additive_operation(self, funcstr, *args, **kwargs):
        """ Generic interface to call function named in funcstr on the data, handling normalisation and reshaping """
        # store data shape, flatten to N x ndim array then restore
        normed = kwargs.pop('normed', True)
        try:
            shp = args[0].shape
        except AttributeError:
            # inputs not arrays
            shp = np.array(args[0], dtype=np.float64).shape
        flat_data = np.vstack([np.array(x, dtype=np.float64).flatten() for x in args]).transpose()

        with closing(mp.Pool(processes=self.ncpu)) as pool:
            z = sum(pool.map(partial(runner_parallel, fstr=funcstr, fd=flat_data), self.kernel_clusters))
            # z = sum(pool.map(partial(runner_debug, fd=flat_data), self.kernel_clusters))

        if normed:
            z /= float(self.ndata)
        return np.reshape(z, shp)

    def pdf(self, *args, **kwargs):
        if len(args) != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")
        return self._additive_operation('pdf', *args, **kwargs)

    def pdf_interp_fn(self, *args, **kwargs):
        """ Return a callable interpolation function based on the grid points supplied in args. """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        flat_data = np.vstack([np.array(x, dtype=np.float64).flatten() for x in args]).transpose()
        # linear interpolation is slower than actually evaluating the KDE, so not worth it:
        # return LinearNDInterpolator(flat_data, self.pdf(*args, **kwargs).flatten())
        return NearestNDInterpolator(flat_data, self.pdf(*args, **kwargs).flatten())

    def marginal_pdf(self, x, **kwargs):
        # return the marginal pdf in the dim specified in kwargs (dim=0 default)
        return self._additive_operation('marginal_pdf', x, **kwargs)

    def marginal_cdf(self, x, **kwargs):
        """ Return the marginal cdf in the dim specified in kwargs (dim=0 default) """
        return self._additive_operation('marginal_cdf', x, **kwargs)

    def marginal_icdf(self, y, *args, **kwargs):
        """ Return value of inverse marginal CDF in specified dim """
        if not 0. < y < 1.:
            raise AttributeError("Input variable y must lie in range (0, 1)")
        dim = kwargs.pop('dim', 0)
        try:
            xopt = marginal_icdf_optimise(self, y, dim=dim, tol=1e-12)
        except Exception as e:
            print "Failed to optimise marginal icdf"
            raise e
        return xopt

    def values_at_data(self, **kwargs):
        return self.pdf(*[self.data[:, i] for i in range(self.ndim)], **kwargs)

    def values_on_grid(self, n_points=10):
        grids = np.meshgrid(*[np.linspace(mi, ma, n_points) for mi, ma in zip(self.min_array, self.max_array)])
        it = np.nditer(grids + [None,])
        for x in it:
            x[-1][...] = self.pdf(*x[:-1]) # NB must unpack, as x[:-1] is a tuple
        return it.operands[-1]

    @property
    def marginal_mean(self):
        return np.mean(self.data, axis=0)

    @property
    def marginal_second_moment(self):
        return np.sum(self.bandwidths ** 2 + self.data ** 2, axis=0) / float(self.ndata)

    @property
    def marginal_variance(self):
        return self.marginal_second_moment - self.marginal_mean ** 2

    def _t_dependent_variance(self, t):
        ## FIXME: BROKEN, and not sufficiently generic since it relies on calling mvns
        ## TODO: test me!
        # have already checked that ndim > 1 by this point
        z0 = np.tile(
            np.array([m.marginal_pdf(t, dim=0) for m in self.mvns]).reshape((self.ndata, 1)),
            (1, self.ndim - 1)
        )
        tdm = np.mean(self.data[:, 1:] * z0, axis=0)
        tdsm = np.mean((self.bandwidths[:, 1:] ** 2 + self.data[:, 1:] ** 2) * z0, axis=0)
        tdv = tdsm - tdm ** 2

        return tdm, tdsm, tdv

    def t_dependent_mean(self, t):
        if self.ndim == 1:
            raise NotImplementedError("Unable to compute time-dependent mean with ndim=1")
        return self._t_dependent_variance(t)[0]

    def t_dependent_second_moment(self, t):
        if self.ndim == 1:
            raise NotImplementedError("Unable to compute time-dependent second moment with ndim=1")
        return self._t_dependent_variance(t)[1]

    def t_dependent_variance(self, t):
        if self.ndim == 1:
            raise NotImplementedError("Unable to compute time-dependent variance with ndim=1")
        return self._t_dependent_variance(t)[2]


def _init(arr_pt_to_populate, shp_to_populate):
    """ Each pool process calls this initializer. Load the array to be populated into that process's global namespace """
    global shp
    global arr_pt
    shp = shp_to_populate
    arr_pt = arr_pt_to_populate


def runner_shared_mem(x, fstr=None):
    global shp
    global arr_pt
    v = np.ctypeslib.as_array(arr_pt, shape=shp)
    # print "runner(x, %s, %s)" % (fstr, str(v))
    return x.additive_operation(fstr, v)


class FixedBandwidthKdeShared(FixedBandwidthKde):

    def _additive_operation(self, funcstr, *args, **kwargs):
        """ Generic interface to call function named in funcstr on the data, handling normalisation and reshaping """
        # store data shape, flatten to N x ndim array then restore

        normed = kwargs.pop('normed', True)
        try:
            shp = args[0].shape
        except AttributeError:
            # inputs not arrays
            shp = np.array(args[0], dtype=np.float64).shape
        flat_data = np.vstack([np.array(x, dtype=np.float64).flatten() for x in args]).transpose()

        # ctypes POINTER
        c_double_p = ctypes.POINTER(ctypes.c_double)
        flat_data_ctypes_p = flat_data.ctypes.data_as(c_double_p)
        # c_float_p = ctypes.POINTER(ctypes.c_float)
        # flat_data_ctypes_p = flat_data.astype(np.float32).ctypes.data_as(c_float_p)

        # ctypes ARRAY
        # concat_data = flat_data.flat
        # flat_data_ctypes = mp.sharedctypes.RawArray('d', concat_data)

        with closing(
                mp.Pool(processes=self.ncpu, initializer=_init, initargs=(flat_data_ctypes_p, flat_data.shape))
        ) as pool:
            z = sum(pool.map(partial(runner_shared_mem, fstr=funcstr), self.kernel_clusters))

        if normed:
            z /= float(self.ndata)
        return np.reshape(z, shp)