__author__ = 'gabriel'
import numpy as np
import operator
from kde import kernels
from scipy.stats import norm
from scipy.spatial import KDTree
from kde.kernels import c3


class FixedBandwidthKde():
    def __init__(self, data, *args, **kwargs):
        self.data = data
        if len(data.shape) == 1:
            self.data = np.array(data).reshape((len(data), 1))

        self.bandwidths = None
        self.set_bandwidths(*args, **kwargs)

    def set_bandwidths(self, *args, **kwargs):

        try:
            bandwidths = kwargs.pop('bandwidths')
        except KeyError:
            bandwidths = self.range_array / float(self.ndata)

        if len(bandwidths) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = np.tile(bandwidths, (self.ndata, 1))
        self.set_mvns()


    def set_mvns(self):
        self.mvns = [kernels.MultivariateNormal(self.data[i], self.bandwidths[i]**2) for i in range(self.ndata)]
        # self.mvns = [c3.MultivariateNormal(self.data[i], self.bandwidths[i]**2) for i in range(self.ndata)]

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

    def pdf(self, *args, **kwargs):
        if len(args) != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")

        tol = kwargs.pop('tol', None)
        if tol is None:
            # no tolerance specified - use all MVNs
            # store data shape, flatten to N x ndim array then restore
            try:
                shp = args[0].shape
            except AttributeError:
                # inputs not arrays
                shp = np.array(args[0], dtype=np.float64).shape
            flat_data = np.vstack([np.array(x, dtype=np.float64).flatten() for x in args]).transpose()
            z = reduce(operator.add, [x.pdf(flat_data) for x in self.mvns]) / float(self.ndata)
            return np.reshape(z, shp)
        else:
            # use specified tolerance to filter MVNs
            std_cutoff = norm.ppf(1 - tol)
            # these values define the multi-dim cuboid in which sources must lie:
            real_cutoffs = std_cutoff * self.bandwidths
            filt = lambda x: np.all(np.abs(self.data - x) < real_cutoffs, axis=1)
            get_mvns = lambda x: np.array(self.mvns)[filt(x)]

        # it = np.nditer(args + (None,))
        it = np.nditer(args + (None,), flags=['buffered'], op_dtypes=['float64'] * (self.ndim + 1), casting='same_kind')
        for x in it:
            # x[self.ndim][...] = np.sum([y.pdf(*x[:-1]) for y in get_mvns(x[:-1])])
            x[self.ndim][...] = np.sum([y.pdf(np.array(x[:-1], dtype=np.float64)) for y in get_mvns(x[:-1])])

        return it.operands[self.ndim] / float(self.ndata)

    def values_at_data(self, **kwargs):
        return self.pdf(*[self.data[:, i] for i in range(self.ndim)], **kwargs)
        # return np.power(2*PI, -self.ndim * 0.5) / np.prod(self.std_devs ** 2, axis=1) / float(self.ndata)

    def values_on_grid(self, n_points=10):
        grids = np.meshgrid(*[np.linspace(mi, ma, n_points) for mi, ma in zip(self.min_array, self.max_array)])
        it = np.nditer(grids + [None,])
        for x in it:
            x[-1][...] = self.pdf(*x[:-1]) # NB must unpack, as x[:-1] is a tuple
        return it.operands[-1]


class VariableBandwidthKdeIsotropic(FixedBandwidthKde):

    def set_bandwidths(self, *args, **kwargs):
        try:
            self.nn = kwargs.pop('nn')
        except KeyError:
            # default values
            self.nn = 10 if self.ndim == 1 else 100

        # compute nn distances on unnormed data
        kd = KDTree(self.data)

        self.bandwidths = np.zeros((self.ndata, self.ndim))

        for i in range(self.ndata):
            d, _ = kd.query(self.data[i, :], k=self.nn)
            nn = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]
            # all dims have same bandwidth
            self.bandwidths[i] = np.ones(self.ndim) * nn
        self.set_mvns()

class VariableBandwidthKde(FixedBandwidthKde):

    def set_bandwidths(self, *args, **kwargs):
        if 'nn' in kwargs:
            self.nn = kwargs.pop('nn')
            self.compute_nn_bandwidth()
        elif 'bandwidths' in kwargs:
            bandwidths = np.array(kwargs.pop('bandwidths'), dtype=float)
            if ( len(bandwidths.shape) == 1 ) and ( self.ndim == 1 ) and ( bandwidths.size == self.ndata ):
                self.bandwidths = bandwidths
            elif ( bandwidths.shape[1] == self.ndim ) and ( bandwidths.shape[0] == self.ndata ):
                self.bandwidths = bandwidths
            else:
                raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")
        else:
            # default nn values
            self.nn = min(10, self.ndata) if self.ndim == 1 else min(100, self.ndata)
            self.compute_nn_bandwidth()

        self.set_mvns()

    def compute_nn_bandwidth(self):
        if self.nn <= 1:
            raise Exception("The number of nearest neighbours for variable KDE must be >1")
        # compute nn distances
        nd = self.normed_data
        std = self.raw_std_devs
        kd = KDTree(nd)
        # kd = KDTree(self.data)

        self.nn_distances = np.zeros(self.ndata)
        self.bandwidths = np.zeros((self.ndata, self.ndim))

        for i in range(self.ndata):
            d, _ = kd.query(nd[i, :], k=self.nn)
            # d, _ = kd.query(self.data[i, :], k=self.nn)
            self.nn_distances[i] = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]
            self.bandwidths[i] = std * self.nn_distances[i]
            # self.bandwidths[i] = np.ones(self.ndim) * self.nn_distances[i]

    @property
    def raw_std_devs(self):
        return np.std(self.data, axis=0)

    @property
    def normed_data(self):
        return self.data / self.raw_std_devs