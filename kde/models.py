__author__ = 'gabriel'
import numpy as np
import operator
from contextlib import closing
from functools import partial
import ctypes
import multiprocessing as mp
from kde import kernels
from stats.logic import weighted_stdev
from sklearn.neighbors import NearestNeighbors
from data.models import Data, DataArray, SpaceTimeDataArray, CartesianSpaceTimeData, negative_time_dimension
import ipdb  # just in case
import warnings

# some utility functions

def marginal_icdf_optimise(k, y, dim=0, tol=1e-8):

    n = 100
    max_iter = 50
    f = lambda x: np.abs(k.marginal_cdf(x, dim=dim) - y)
    mean_bd = np.mean(k.bandwidths[:, dim])
    minx = np.min(k.data[:, dim])
    maxx = np.max(k.data[:, dim])
    err = 1.
    niter = 0
    x0 = 0.
    while err > tol:
        if niter > max_iter:
            raise Exception("Failed to converge to optimum after %u iterations", max_iter)
        xe = DataArray(np.linspace(minx, maxx, n))
        ye = f(xe)
        idx = np.argmin(ye)
        if idx == 0:
            # return xe[idx]
            minx -= mean_bd
            continue
        if idx == (n-1):
            maxx += mean_bd
            continue
        err = ye[idx]
        x0 = xe[idx]
        minx = xe[idx - 1]
        maxx = xe[idx + 1]
        niter += 1
    return float(x0)


def set_nn_bandwidths(normed_data, raw_stdevs, num_nn, **kwargs):
    tol = 1e-12

    min_bandwidth = kwargs.get('min_bandwidth', None)

    # compute nn distances on normed data

    from time import time
    tic = time()
    try:
        nn_obj = NearestNeighbors(num_nn).fit(normed_data)
        dist, _ = nn_obj.kneighbors(normed_data)
    except ValueError as exc:
        ## should never end up here as num_nn is set at instantiation
        ipdb.set_trace()
        warnings.warn("Insufficient data, reducing number of nns to %d" % normed_data.ndata)
        nn_obj = NearestNeighbors(normed_data.ndata).fit(normed_data)
        dist, _ = nn_obj.kneighbors(normed_data)
    except Exception as exc:
        ipdb.set_trace()

    nn_distances = dist[:, -1]

    if np.any(np.isinf(nn_distances)):
        raise AttributeError("Encountered np.inf values in NN distances")

    # check for NN distances below tolerance
    intol_idx = np.where(nn_distances < tol)[0]
    if len(intol_idx):
        intol_data = normed_data[intol_idx]
        nn_obj.n_neighbors = normed_data.ndata
        dist, _ = nn_obj.kneighbors(intol_data)
        # for each point, perform a NN distance lookup on ALL valid NNs and use the first one above tol
        for i, j in enumerate(intol_idx):
            d = dist[i][num_nn + 1:]
            nn_distances[j] = d[d > tol][0]

    nn_distances = nn_distances.reshape((normed_data.ndata, 1))
    bandwidths = raw_stdevs * nn_distances

    # apply minimum bandwidth constraint if required
    if min_bandwidth is not None and np.any(bandwidths < min_bandwidth):
        fix_idx = np.where(bandwidths < min_bandwidth)
        bandwidths[fix_idx] = np.array(min_bandwidth)[fix_idx[1]]

    return nn_distances, bandwidths

# A few helper functions required for parallel processing


def runner_additive(x, fstr=None, fd=None, **kwargs):
    return x.additive_operation(fstr, fd, **kwargs)


def runner_additive_shared(x, fstr=None, **kwargs):
    global shp
    global arr_pt
    v = np.ctypeslib.as_array(arr_pt, shape=shp)
    return x.additive_operation(fstr, v, **kwargs)


def runner(x, fstr=None, fd=None, **kwargs):
    return x.operation(fstr, fd, **kwargs)


def runner_shared(x, fstr=None, **kwargs):
    global shp
    global arr_pt
    v = np.ctypeslib.as_array(arr_pt, shape=shp)
    return x.operation(fstr, v, **kwargs)


def shared_process_init(arr_pt_to_populate, shp_to_populate):
    """ Each pool process calls this initializer. Load the array to be populated into that process's global namespace """
    global shp
    global arr_pt
    shp = shp_to_populate
    arr_pt = arr_pt_to_populate


class KernelCluster(object):
    """ Class for holding a 'cluster' of kernels, useful for parallelisation. """

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

    def iter_operate(self, funcstr, data, **kwargs):
        """
        Return an iterator that executes the fun named in funcstr to each kernel in turn.
        The function is passed the data and any kwargs supplied.
        """
        return (getattr(x, funcstr)(data, **kwargs) for x in self.kernels)

    def additive_operation(self, funcstr, data, **kwargs):
        """ Generic interface to call function named in funcstr on the data, reducing data by summing """
        return reduce(operator.add, self.iter_operate(funcstr, data, **kwargs))

    def operation(self, funcstr, data, **kwargs):
        return list(self.iter_operate(funcstr, data, **kwargs))


class WeightedKernelCluster(KernelCluster):

    def __init__(self, data, weights, bandwidths, ktype=None):
        self.weights = weights
        super(WeightedKernelCluster, self).__init__(data, bandwidths, ktype=ktype)

    def iter_operate(self, funcstr, data, **kwargs):
        """
        Return an iterator that executes the fun named in funcstr to each kernel in turn.
        The function is passed the data and any kwargs supplied.
        """
        return (w * getattr(x, funcstr)(data, **kwargs) for (w, x) in zip(self.weights, self.kernels))


class KdeBase(object):

    kernel_class = kernels.BaseKernel
    data_class = DataArray

    def __init__(self, data, parallel=True, *args, **kwargs):
        if isinstance(data, self.data_class):
            self.data = data
        else:
            self.data = self.data_class(data)

        self.parallel = parallel

        try:
            self.ncpu = kwargs.pop('ncpu', mp.cpu_count())
        except NotImplementedError:
            self.ncpu = 1
        self.b_shared = kwargs.pop('sharedmem', False)

        self.kernel_clusters = None
        self.bandwidths = None
        self.set_bandwidths(*args, **kwargs)
        self.set_kernels()

    @property
    def ndim(self):
        return self.data.nd

    @property
    def ndata(self):
        return self.data.ndata

    @property
    def raw_std_devs(self):
        return np.std(self.data, axis=0, ddof=1)

    def set_bandwidths(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def distributed_indices(self):
        """
        :return: List of (start, end) indices into data, bandwidths, each corresponding to the tasks of a worker
        """
        if not self.parallel:
            return [(0, self.ndata)]

        ## TODO: check this cutoff value
        if self.ndata < (self.ncpu * 50):
            # not worth splitting data up
            return [(0, self.ndata)]

        n_per_cluster = int(self.ndata / self.ncpu)
        indices = []
        idx = 0
        for i in range(self.ncpu - 1):
            indices.append((idx, idx + n_per_cluster))
            idx += n_per_cluster
        indices.append((idx, self.ndata))
        return indices

    def set_kernels(self):
        self.kernel_clusters = []
        for i, j in self.distributed_indices:
            this_data = self.data[i:j]
            this_bandwidths = self.bandwidths[i:j]
            self.kernel_clusters.append(KernelCluster(this_data, this_bandwidths, ktype=self.kernel_class))

    def _iterative_operation(self, funcstr, target, *args, **kwargs):
        """
        Generic interface to call function named in funcstr on the target data
        The returned list contains an element for each kernel
        """

        if self.b_shared:

            # create ctypes array pointer
            c_double_p = ctypes.POINTER(ctypes.c_double)
            flat_data_ctypes_p = target.ctypes.data_as(c_double_p)
            with closing(
                    mp.Pool(processes=self.ncpu, initializer=shared_process_init,
                            initargs=(flat_data_ctypes_p, target.shape))
            ) as pool:
                z = pool.map(partial(runner_shared, fstr=funcstr), self.kernel_clusters)

        else:
            with closing(mp.Pool(processes=self.ncpu)) as pool:
                z = pool.map(partial(runner, fstr=funcstr, fd=target), self.kernel_clusters)

        return reduce(operator.add, z)

    def _additive_operation(self, funcstr, target, **kwargs):
        """ Generic interface to call function named in funcstr on the target data, handling normalisation """

        normed = kwargs.pop('normed', True)

        if not self.parallel:
            z = sum([x.additive_operation(funcstr, target, **kwargs) for x in self.kernel_clusters])
            # z = self.kernel_clusters[0].additive_operation(funcstr, target, **kwargs)
        elif self.b_shared:
            # create ctypes array pointer
            c_double_p = ctypes.POINTER(ctypes.c_double)
            flat_data_ctypes_p = target.ctypes.data_as(c_double_p)
            with closing(
                    mp.Pool(processes=self.ncpu, initializer=shared_process_init,
                            initargs=(flat_data_ctypes_p, target.shape))
            ) as pool:
                z = sum(pool.map(partial(runner_additive_shared, fstr=funcstr, **kwargs), self.kernel_clusters))
        else:
            with closing(mp.Pool(processes=self.ncpu)) as pool:
                z = sum(pool.map(partial(runner_additive, fstr=funcstr, fd=target, **kwargs), self.kernel_clusters))

        if normed:
            z /= float(self.ndata)
        return z

    def check_inputs(self, x, ndim=None, cls=None):
        ndim = ndim or self.ndim
        cls = cls or self.data_class
        if not isinstance(x, cls):
            x = cls(x)

        if x.nd != ndim:
            raise AttributeError("Target data does not have the correct number of dimensions")

        return x

    def pdf(self, target, **kwargs):
        target = self.check_inputs(target, ndim=self.ndim)
        return self._additive_operation('pdf', target, **kwargs)

    def marginal_pdf(self, x, **kwargs):
        # return the marginal pdf in the dim specified in kwargs (dim=0 default)
        x = self.check_inputs(x, ndim=1, cls=DataArray)
        return self._additive_operation('marginal_pdf', x, **kwargs)

    def marginal_cdf(self, x, **kwargs):
        """ Return the marginal cdf in the dim specified in kwargs (dim=0 default) """
        x = self.check_inputs(x, ndim=1, cls=DataArray)
        return self._additive_operation('marginal_cdf', x, **kwargs)

    def partial_marginal_pdf(self, x, **kwargs):
        # return the marginal pdf in all dims but the one specified in kwargs (dim=0 by default)
        x = self.check_inputs(x, ndim=self.ndim - 1, cls=DataArray)
        return self._additive_operation('partial_marginal_pdf', x, **kwargs)

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


class KdeBaseSeparable(KdeBase):
    """
    KDE that is separable in time and space.  Requires the use of SpaceTimeData class to tease apart separable dims
    """
    data_class = SpaceTimeDataArray

    def pdf(self, target, **kwargs):
        self.check_inputs(target, ndim=self.ndim)

        # normed kwarg is treated as a special case here
        # if True, BOTH parts need norming
        # if False, ONE part needs norming
        # in practice, best just to ALWAYS norm the spatial component

        marg = self.marginal_pdf(target.time, **kwargs)
        kwargs['normed'] = True
        pmarg = self.partial_marginal_pdf(target.space, **kwargs)

        return marg * pmarg

    def partial_marginal_pdf(self, x, **kwargs):
        # return the marginal pdf in all dims but the one specified in kwargs (dim=0 by default)
        self.check_inputs(x, ndim=self.ndim - 1)
        dim = kwargs.get('dim')
        if dim and dim != 0:
            raise NotImplementedError("Unsupported operation: partial_marginal_pdf with dim != 0")
        return self._additive_operation('partial_marginal_pdf', x, **kwargs)


class FixedBandwidthKde(KdeBase):
    kernel_class = kernels.MultivariateNormal

    def set_bandwidths(self, *args, **kwargs):
        bandwidths = kwargs.pop('bandwidths')

        if not hasattr(bandwidths, '__iter__'):
            bandwidths = [bandwidths] * self.ndim

        if len(bandwidths) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = np.tile(bandwidths, (self.ndata, 1))

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
        ## TODO: test me!
        # have already checked that ndim > 1 by this point

        z0 = np.tile(
            np.array(self._iterative_operation('marginal_pdf', t, dim=0)).reshape((self.ndata, 1)),
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


class FixedBandwidthKdeSeparable(FixedBandwidthKde, KdeBaseSeparable):
    """
    Combination of fixed bandwidth and separable KDE in time / space.
    """


class VariableBandwidthKde(FixedBandwidthKde):

    def set_bandwidths(self, *args, **kwargs):
        bandwidths = np.array(kwargs.pop('bandwidths'), dtype=float)

        if bandwidths.ndim == 1:
            if (self.ndim == 1) and (bandwidths.size == self.ndata):
                self.bandwidths = bandwidths.reshape((self.ndata, 1))
            else:
                raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        elif (bandwidths.shape[1] == self.ndim) and (bandwidths.shape[0] == self.ndata):
            self.bandwidths = bandwidths

        else:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

    @property
    def normed_data(self):
        return self.data / self.raw_std_devs


class VariableBandwidthNnKde(VariableBandwidthKde):

    def __init__(self, data, *args, **kwargs):

        self.nn = kwargs.pop('number_nn')
        if self.nn < 1:
            raise AttributeError("The number of nearest neighbours for variable KDE must be >=1")

        self.nn_distances = []
        super(VariableBandwidthNnKde, self).__init__(data, *args, **kwargs)

        # check requested number NN if supplied.
        if (self.nn is not None) and (self.nn > self.ndata):
            warnings.warn("Requested number of NNs (%d) is too large for the size of the dataset (%d)"
                                 % (self.nn, self.ndata))
            self.nn = self.ndata

    def set_bandwidths(self, *args, **kwargs):
        # check number of datapoints > 1
        if self.ndata <= 1:
            raise AttributeError("2 or more datapoints required for variable bandwidth KDE")
        self.nn_distances, self.bandwidths = set_nn_bandwidths(self.normed_data, self.raw_std_devs, self.nn, **kwargs)


class WeightedFixedBandwidthKde(FixedBandwidthKde):
    def __init__(self, data, weights, *args, **kwargs):
        self.weights = np.array(weights)
        super(WeightedFixedBandwidthKde, self).__init__(data, *args, **kwargs)

    def set_kernels(self):
        self.kernel_clusters = []
        for i, j in self.distributed_indices:
            this_data = self.data[i:j]
            this_bandwidths = self.bandwidths[i:j]
            this_weights = self.weights[i:j]
            self.kernel_clusters.append(
                WeightedKernelCluster(this_data, this_weights, this_bandwidths, ktype=self.kernel_class)
            )

    @property
    def raw_std_devs(self):
        # weighted standard deviation calculation
        return weighted_stdev(self.data.data, self.weights)

    def set_weights(self, weights):
        """ Required so that bandwidths are recomputed upon switching weights """
        self.weights = weights
        self.set_bandwidths()

    def _iterative_operation(self, funcstr, *args, **kwargs):
        raise NotImplementedError()


class WeightedVariableBandwidthKde(WeightedFixedBandwidthKde):

    def set_bandwidths(self, *args, **kwargs):
        bandwidths = kwargs.pop('bandwidths')

        if not isinstance(bandwidths, np.ndarray):
            bandwidths = np.array(bandwidths)

        if len(bandwidths) != self.ndata:
            raise AttributeError("Number of supplied bandwidths does not match the number of datapoints")

        if len(bandwidths[0]) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = bandwidths


class WeightedVariableBandwidthNnKde(VariableBandwidthNnKde, WeightedFixedBandwidthKde):
    pass


class FixedBandwidthXValidationKde(FixedBandwidthKde):

    def set_bandwidths(self, *args, **kwargs):
        self.xvfold = kwargs.pop('xvfold', min(20, self.ndata))
        self.compute_xv_bandwidth()

    def compute_xv_bandwidth(self, hmin=None, hmax=None):
        from itertools import combinations
        from scipy import optimize
        from copy import copy

        # define a range for the bandwidth
        hmin = hmin or 0.1 # 1/10 x standard deviation
        hmax = hmax or 5 # 5 x standard deviation

        # shuffle data indices
        idx = np.random.permutation(self.ndata)

        # select xvfold validation sets
        validation_sets = [idx[i::self.xvfold] for i in range(self.xvfold)]
        def training_idx_gen():
            # yields (testing idx, training idx) tuples
            for i in range(self.xvfold):
                yield validation_sets[i], np.concatenate(validation_sets[:i] + validation_sets[(i+1):])

        # CV score for minimisation
        def cv_score(h):
            ll = 0.0 # log likelihood
            for test_idx, train_idx in training_idx_gen():
                testing_data = self.data[test_idx, :]
                training_data = self.data[train_idx, :]
                new_kde = FixedBandwidthKde(training_data, bandwidths=self.raw_std_devs * h, parallel=False)
                res = new_kde.pdf(*testing_data.transpose())
                ll += np.sum(np.log(res))
            return -ll / float(self.xvfold)

        res = optimize.minimize_scalar(cv_score, bounds=[hmin, hmax],
                                       method='bounded', options={'disp': True})
        if res.success:
            self.bandwidths = np.tile(self.raw_std_devs * res.x, (self.ndata, 1))
        else:
            raise ValueError("Unable to find max likelihood bandwidth")


class VariableBandwidthNnKdeSeparable(FixedBandwidthKde, KdeBaseSeparable):

    def __init__(self, data, *args, **kwargs):

        self.nn = kwargs.pop('number_nn')
        if not hasattr(self.nn, '__iter__'):
            self.nn = [self.nn, self.nn]
        self.nn_distances_t = []
        self.nn_distances_s = []

        super(VariableBandwidthNnKdeSeparable, self).__init__(data, *args, **kwargs)

        if len(self.nn) != 2:
            raise AttributeError("Separable KDE accepts TWO nearest neighbour numbers")

        for n in self.nn:
            if n < 1:
                raise AttributeError("The number of nearest neighbours for variable KDE must be >=1")
            if n > (self.ndata - 1):
                raise AttributeError(
                    "Requested number of NNs (%d) is too large for the size of the dataset (%d)" % (n, self.ndata)
                )

    def set_bandwidths(self, *args, **kwargs):
        # set bandwidths separately in the separable dimensions

        # time
        self.nn_distances_t, bandwidths_t = set_nn_bandwidths(self.data.time / self.raw_std_devs[0],
                                                         self.raw_std_devs[0],
                                                         self.nn[0], **kwargs)
        # space
        self.nn_distances_s, bandwidths_s = set_nn_bandwidths(self.data.space / self.raw_std_devs[1:],
                                                         self.raw_std_devs[1:],
                                                         self.nn[1], **kwargs)
        self.bandwidths = np.hstack((bandwidths_t, bandwidths_s))


class WeightedVariableBandwidthNnKdeSeparable(WeightedFixedBandwidthKde, VariableBandwidthNnKdeSeparable):
    pass


class SpaceTimeExponentialVariableBandwidthNnKde(VariableBandwidthNnKde):
    kernel_class = kernels.SpaceNormalTimeExponential


class SpaceTimeVariableBandwidthNnTimeReflected(VariableBandwidthNnKde):

    data_class = SpaceTimeDataArray
    kernel_class = kernels.SpaceTimeNormalReflective


class SpaceTimeVariableBandwidthNnTimeOneSided(VariableBandwidthNnKde):

    data_class = SpaceTimeDataArray
    kernel_class = kernels.SpaceTimeNormalOneSided
