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
import ipdb  # just in case

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
        xe = np.linspace(minx, maxx, n)
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
    return x0


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

    def __init__(self, data, parallel=True, *args, **kwargs):
        # print "KdeBase.__init__"
        self.data = np.array(data)
        if self.data.ndim == 1:
            self.data = self.data.reshape((self.data.size, 1))

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
        return self.data.shape[1]

    @property
    def ndata(self):
        return self.data.shape[0]

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

    def _iterative_operation(self, funcstr, *args, **kwargs):
        """
        Generic interface to call function named in funcstr on the data, handling normalisation and reshaping
        The returned list contains an element for each kernel
        """
        ## TODO: this is experimental, may be better to return a FLATTENED array and a shape?
        try:
            shp = args[0].shape
        except AttributeError:
            # inputs not arrays
            shp = np.array(args[0], dtype=np.float64).shape
        flat_data = np.vstack([np.array(x, dtype=np.float64).flatten() for x in args]).transpose()

        if self.b_shared:

            # create ctypes array pointer
            c_double_p = ctypes.POINTER(ctypes.c_double)
            flat_data_ctypes_p = flat_data.ctypes.data_as(c_double_p)
            with closing(
                    mp.Pool(processes=self.ncpu, initializer=shared_process_init,
                            initargs=(flat_data_ctypes_p, flat_data.shape))
            ) as pool:
                z = pool.map(partial(runner_shared, fstr=funcstr), self.kernel_clusters)

        else:
            with closing(mp.Pool(processes=self.ncpu)) as pool:
                z = pool.map(partial(runner, fstr=funcstr, fd=flat_data), self.kernel_clusters)

        return reduce(operator.add, [[x.reshape(shp) for x in y] for y in z])

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

        if not self.parallel:
            z = self.kernel_clusters[0].additive_operation(funcstr, flat_data, **kwargs)
        elif self.b_shared:
            # create ctypes array pointer
            c_double_p = ctypes.POINTER(ctypes.c_double)
            flat_data_ctypes_p = flat_data.ctypes.data_as(c_double_p)
            with closing(
                    mp.Pool(processes=self.ncpu, initializer=shared_process_init,
                            initargs=(flat_data_ctypes_p, flat_data.shape))
            ) as pool:
                z = sum(pool.map(partial(runner_additive_shared, fstr=funcstr, **kwargs), self.kernel_clusters))
        else:
            with closing(mp.Pool(processes=self.ncpu)) as pool:
                z = sum(pool.map(partial(runner_additive, fstr=funcstr, fd=flat_data, **kwargs), self.kernel_clusters))

        if normed:
            z /= float(self.ndata)
        return np.reshape(z, shp)

    def pdf(self, *args, **kwargs):
        if len(args) != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")
        return self._additive_operation('pdf', *args, **kwargs)

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
        # z0 = np.tile(
        #     np.array([m.marginal_pdf(t, dim=0) for m in self.mvns]).reshape((self.ndata, 1)),
        #     (1, self.ndim - 1)
        # )
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
        # print "VariableBandwidthNnKde.__init__"
        # print args
        # print kwargs

        self.nn = kwargs.pop('nn', None)
        self.nn_distances = []
        super(VariableBandwidthNnKde, self).__init__(data, *args, **kwargs)
        # check requested number NN if supplied.
        if (self.nn is not None) and (self.nn > (self.ndata - 1)):
            raise AttributeError("Requested number of NNs (%d) is too large for the size of the dataset (%d)"
                                 % (self.nn, self.ndata))

    def set_bandwidths(self, *args, **kwargs):
        tol = 1e-12

        default_distance = kwargs.pop('nn_default_distance', None)
        min_bandwidth = kwargs.pop('min_bandwidth', None)
        if not self.nn:
            # default nn values
            self.nn = min(100, self.ndata) if self.ndim == 1 else min(15, self.ndata)

        if self.nn <= 1:
            raise AttributeError("The number of nearest neighbours for variable KDE must be >1")

        # compute nn distances on normed data
        nd = self.normed_data
        std = self.raw_std_devs

        # increment NN by one since first result is always self-match
        from time import time
        tic = time()
        try:
            nn_obj = NearestNeighbors(self.nn + 1).fit(nd)
            dist, _ = nn_obj.kneighbors(nd)
        except Exception as exc:
            ipdb.set_trace()
        print "NN computation complete in %f s" % (time() - tic)

        self.nn_distances = dist[:, -1]

        if np.any(np.isinf(self.nn_distances)):
            raise AttributeError("Encountered np.inf values in NN distances")

        # check for NN distances below tolerance
        intol_idx = np.where(self.nn_distances < tol)[0]
        if len(intol_idx):
            intol_data = nd[intol_idx]
            nn_obj.n_neighbors = self.ndata
            dist, _ = nn_obj.kneighbors(intol_data)
            # for each point, perform a NN distance lookup on ALL valid NNs and use the first one above tol
            for i, j in enumerate(intol_idx):
                d = dist[i][self.nn + 1:]
                self.nn_distances[j] = d[d > tol][0]

        self.nn_distances = self.nn_distances.reshape((self.ndata, 1))
        self.bandwidths = std * self.nn_distances

        # apply minimum bandwidth constraint if required
        if min_bandwidth is not None and np.any(self.bandwidths < min_bandwidth):
            fix_idx = np.where(self.bandwidths < min_bandwidth)
            self.bandwidths[fix_idx] = np.array(min_bandwidth)[fix_idx[1]]


class WeightedFixedBandwidthKde(FixedBandwidthKde):
    def __init__(self, data, weights, *args, **kwargs):
        self.weights = np.array(weights)
        super(WeightedFixedBandwidthKde, self).__init__(data, *args, **kwargs)
        # print "WeightedFixedBandwidthKde.__init__"
        # print weights
        # print args
        # print kwargs

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
        return weighted_stdev(self.data, self.weights)

    def set_weights(self, weights):
        """ Required so that bandwidths are recomputed upon switching weights """
        self.weights = weights
        self.set_bandwidths()

    def _iterative_operation(self, funcstr, *args, **kwargs):
        raise NotImplementedError()


class WeightedVariableBandwidthNnKde(VariableBandwidthNnKde, WeightedFixedBandwidthKde):
    pass


class FixedBandwidthXValidationKde(FixedBandwidthKde):

    ## FIXME: not working or finished?

    def set_bandwidths(self, *args, **kwargs):
        self.xvfold = kwargs.pop('xvfold', min(20, self.ndata))
        self.compute_xv_bandwidth()
        self.set_mvns()

    def compute_xv_bandwidth(self, hmin=None, hmax=None):
        from itertools import combinations
        from scipy import optimize
        # define a range for the bandwidth
        hmin = hmin or 0.1 # 1/10 x standard deviation
        hmax = hmax or 5 # 5 x standard deviation

        idx = np.random.permutation(self.ndata)
        idx_sets = [idx[i::self.xvfold] for i in range(self.xvfold)]

        # CV score for minimisation
        def cv_score(h):
            bandwidths = self.raw_std_devs * h
            all_mvns = np.array([kernels.MultivariateNormal(self.data[i], bandwidths**2) for i in range(self.ndata)])
            ll = 0.0 # log likelihood
            idx_sets_excl = combinations(idx_sets, self.xvfold - 1)
            for i, test_idx_set in enumerate(idx_sets_excl):
                testing_data = self.data[idx_sets[i], :]
                training_mvns = all_mvns[np.concatenate(test_idx_set)]
                ll += np.sum([np.log(x.pdf(testing_data)) for x in training_mvns])
            return -ll / float(self.xvfold)

        # minimise CV function over h
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: x - hmin,
            },
            {
                'type': 'ineq',
                'fun': lambda x: hmax - x,
            },
        ]
        res = optimize.minimize(cv_score, [1.0, ], method='L-BFGS-B', constraints=constraints)
        if res.success:
            self.bandwidths = np.tile(self.raw_std_devs * res.x, (self.ndata, 1))
        else:
            raise ValueError("Unable to find max likelihood bandwidth")
