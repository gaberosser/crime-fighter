__author__ = 'gabriel'
import numpy as np
import operator
from kde import kernels


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


def weighted_stdev(data, weights):
    ndata = data.shape[0]
    if weights.ndim != 1:
        raise AttributeError("Weights must be a 1D array")
    if weights.size != ndata:
        raise AttributeError("Length of weights vector not equal to number of data")

    if data.ndim == 1:
        # 1D data
        ndim = 1
        _data = data.reshape(ndata, 1)
    else:
        ndim = data.shape[1]
        _data = data

    tiled_weights = np.tile(weights.reshape(ndata, 1), (1, ndim))
    sum_weights = np.sum(weights)
    M = float(sum(weights != 0.))  # number nonzero weights
    wm = np.sum(tiled_weights * _data, axis=0) / sum_weights  # weighted arithmetic mean
    a = np.sum(tiled_weights * ((_data - wm) ** 2), axis=0)  # numerator
    b = sum_weights * (M - 1.) / M  # denominator

    # check return type
    if ndim == 1:
        return np.sqrt(a / b)[0]
    else:
        return np.sqrt(a / b)


class FixedBandwidthKde(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        if len(data.shape) == 1:
            self.data = np.array(data).reshape((len(data), 1))

        self.bandwidths = None
        self.set_bandwidths(*args, **kwargs)
        self.job_server = None
        self.pool = None

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
        # better to use a generator here to reduce memory usage:
        z = reduce(operator.add, (getattr(x, funcstr)(flat_data, **kwargs) for x in self.mvns))
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


class VariableBandwidthKde(FixedBandwidthKde):

    def set_bandwidths(self, *args, **kwargs):
        if 'bandwidths' in kwargs:
            bandwidths = np.array(kwargs.pop('bandwidths'), dtype=float)
            if ( len(bandwidths.shape) == 1 ) and ( self.ndim == 1 ) and ( bandwidths.size == self.ndata ):
                self.bandwidths = bandwidths
            elif ( bandwidths.shape[1] == self.ndim ) and ( bandwidths.shape[0] == self.ndata ):
                self.bandwidths = bandwidths
            else:
                raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")
        else:
            raise AttributeError("Class instantiation requires a supplied bandwidth kwarg.")

        self.set_mvns()

    @property
    def normed_data(self):
        return self.data / self.raw_std_devs


class VariableBandwidthNnKde(VariableBandwidthKde):

    def __init__(self, data, *args, **kwargs):
        self.nn = kwargs.pop('nn', None)
        self.nn_distances = []
        super(VariableBandwidthNnKde, self).__init__(data, *args, **kwargs)
        self.set_mvns()

    def set_bandwidths(self, *args, **kwargs):
        tol = 1e-12
        from scipy.spatial import KDTree
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

        kd = KDTree(nd)

        self.nn_distances = np.zeros(self.ndata)
        self.bandwidths = np.zeros((self.ndata, self.ndim))

        for i in range(self.ndata):
            d, _ = kd.query(nd[i, :], k=self.nn)
            self.nn_distances[i] = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]

            # check for zero distances
            if self.nn_distances[i] < tol:
                if default_distance:
                    self.nn_distances[i] = default_distance
                else:
                    d, _ = kd.query(nd[i, :], k=kd.n) # all NN distance values
                    idx = (~np.isinf(d)) & (d > tol)
                    if np.any(idx):
                        self.nn_distances[i] = d[np.where(idx)[0][0]]
                    else:
                        raise AttributeError("No non-zero and finite NN distances available, and no default specified")

            self.bandwidths[i] = std * self.nn_distances[i]

        # apply minimum bandwidth constraint if required
        if min_bandwidth is not None and np.any(self.bandwidths < min_bandwidth):
            fix_idx = np.where(self.bandwidths < min_bandwidth)
            rep_min = np.tile(min_bandwidth, (self.ndata, 1))
            self.bandwidths[fix_idx] = rep_min[fix_idx]


class WeightedFixedBandwidthKde(FixedBandwidthKde):
    def __init__(self, data, weights, *args, **kwargs):
        self.weights = np.array(weights)
        super(WeightedFixedBandwidthKde, self).__init__(data, *args, **kwargs)

    @property
    def raw_std_devs(self):
        # weighted standard deviation calculation
        return weighted_stdev(self.data, self.weights)

    def set_weights(self, weights):
        """ Required so that bandwidths are recomputed upon switching weights """
        self.weights = weights
        self.set_bandwidths()

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
        # better to use a generator here to reduce memory usage:
        z = reduce(operator.add, (w * getattr(x, funcstr)(flat_data, **kwargs) for w, x in zip(self.weights, self.mvns)))
        if normed:
            z /= sum(self.weights)
        return np.reshape(z, shp)


class WeightedVariableBandwidthNnKde(VariableBandwidthNnKde):
    def __init__(self, data, weights, *args, **kwargs):
        self.weights = np.array(weights)
        super(WeightedVariableBandwidthNnKde, self).__init__(data, *args, **kwargs)

    @property
    def raw_std_devs(self):
        # weighted standard deviation calculation
        return weighted_stdev(self.data, self.weights)

    def set_weights(self, weights):
        """ Required so that bandwidths are recomputed upon switching weights """
        self.weights = weights
        self.set_bandwidths()

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
        # better to use a generator here to reduce memory usage:
        z = reduce(operator.add, (w * getattr(x, funcstr)(flat_data, **kwargs) for w, x in zip(self.weights, self.mvns)))
        if normed:
            z /= sum(self.weights)
        try:
            return np.reshape(z, shp)
        except Exception:
            import ipdb; ipdb.set_trace()


class FixedBandwidthXValidationKde(FixedBandwidthKde):

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