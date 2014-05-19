__author__ = 'gabriel'
import numpy as np
import math
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
            return xe[idx]
        if idx == (n-1):

            maxx += mean_bd
            continue
        err = ye[idx]
        x0 = xe[idx]
        minx = xe[idx - 1]
        maxx = xe[idx + 1]
        niter += 1
    return x0


class FixedBandwidthKde():
    def __init__(self, data, normed=True, *args, **kwargs):
        self.data = data
        self.normed = normed
        if len(data.shape) == 1:
            self.data = np.array(data).reshape((len(data), 1))

        self.bandwidths = None
        self.set_bandwidths(*args, **kwargs)
        self.job_server = None
        self.pool = None
        # if kwargs.pop('parallel', False):
        #     self.start_job_server()

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
        return np.std(self.data, axis=0)

    def _additive_operation(self, funcstr, normed, *args, **kwargs):
        """ Generic interface to call function named in funcstr on the data, handling normalisation and reshaping """
        # store data shape, flatten to N x ndim array then restore
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
        return self._additive_operation('pdf', self.normed, *args, **kwargs)

    def pdf_interp_fn(self, *args, **kwargs):
        """ Return a callable interpolation function based on the grid points supplied in args. """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        flat_data = np.vstack([np.array(x, dtype=np.float64).flatten() for x in args]).transpose()
        # linear interpolation is slower than actually evaluating the KDE, so not worth it:
        # return LinearNDInterpolator(flat_data, self.pdf(*args, **kwargs).flatten())
        return NearestNDInterpolator(flat_data, self.pdf(*args, **kwargs).flatten())

    def marginal_pdf(self, x, **kwargs):
        # return the marginal pdf in the dim specified in kwargs (dim=0 default)
        return self._additive_operation('marginal_pdf', self.normed, x, **kwargs)

    def marginal_cdf(self, x, **kwargs):
        """ Return the marginal cdf in the dim specified in kwargs (dim=0 default) """
        return self._additive_operation('marginal_cdf', True, x, **kwargs)

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

    def set_bandwidths(self, *args, **kwargs):
        if 'nn' in kwargs:
            self.nn = kwargs.pop('nn')
            self.compute_nn_bandwidth()
        else:
            # default nn values
            self.nn = min(100, self.ndata) if self.ndim == 1 else min(15, self.ndata)
            self.compute_nn_bandwidth()

        self.set_mvns()

    def compute_nn_bandwidth(self):
        from scipy.spatial import KDTree
        if self.nn <= 1:
            raise Exception("The number of nearest neighbours for variable KDE must be >1")
        # compute nn distances on normed data
        nd = self.normed_data
        std = self.raw_std_devs
        kd = KDTree(nd)

        self.nn_distances = np.zeros(self.ndata)
        self.bandwidths = np.zeros((self.ndata, self.ndim))

        for i in range(self.ndata):
            d, _ = kd.query(nd[i, :], k=self.nn)
            self.nn_distances[i] = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]
            self.bandwidths[i] = std * self.nn_distances[i]


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
