__author__ = 'gabriel'
import numpy as np
# import simulate
# from scipy.spatial import KDTree
# from scipy.stats import norm, multivariate_normal
# import operator
PI = np.pi


# class MultivariateNormal():
#
#     def __init__(self, mean, vars):
#         self.ndim = len(vars)
#         self.mean = mean
#         self.vars = vars
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
#             x[self.ndim][...] = self.normnd(it[:self.ndim], self.mean, self.vars)
#             # x[self.ndim][...] = np.prod([self.norm1d(it[i], self.mean[i], self.vars[i]) for i in range(self.ndim)])
#
#         return it.operands[self.ndim]
#
#     @staticmethod
#     def norm1d(x, mu, var):
#         return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))
#
#     def normnd(self, x, mu, var):
#         # each input is a (1 x self.ndim) array
#         a = np.power(2 * PI, self.ndim/2.)
#         b = np.prod(np.sqrt(var))
#         c = -np.sum((x - mu)**2 / (2 * var))
#         return np.exp(c) / (a * b)
#
#
# class MultivariateNormalScipy():
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
#             x[self.ndim][...] = multivariate_normal.pdf(it[:-1], mean=self.mean, cov=self.cov)
#
#         return it.operands[self.ndim]
#
#
# class FixedBandwidthKde():
#     def __init__(self, data, *args, **kwargs):
#         self.data = data
#         if len(data.shape) == 1:
#             self.data = np.array(data).reshape((len(data), 1))
#
#         self.bandwidths = None
#         self.mvns = []
#         self.set_bandwidths(*args, **kwargs)
#
#     def set_bandwidths(self, *args, **kwargs):
#
#         try:
#             bandwidths = kwargs.pop('bandwidths')
#         except KeyError:
#             bandwidths = self.range_array / float(self.ndata)
#
#         if len(bandwidths) != self.ndim:
#             raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")
#
#         self.bandwidths = np.tile(bandwidths, (self.ndata, 1))
#         self.mvns = [MultivariateNormal(self.data[i], self.bandwidths[i]**2) for i in range(self.ndata)]
#
#     @property
#     def ndim(self):
#         return self.data.shape[1]
#
#     @property
#     def ndata(self):
#         return self.data.shape[0]
#
#     @property
#     def max_array(self):
#         return np.max(self.data, axis=0)
#
#     @property
#     def min_array(self):
#         return np.min(self.data, axis=0)
#
#     @property
#     def range_array(self):
#         return self.max_array - self.min_array
#
#     def pdf(self, *args, **kwargs):
#         if len(args) != self.ndim:
#             raise AttributeError("Incorrect dimensions for input variable")
#
#         tol = kwargs.pop('tol', None)
#         if tol is None:
#             # no tolerance specified - use all MVNs
#             return reduce(operator.add, [x.pdf(*args) for x in self.mvns]) / float(self.ndata)
#         else:
#             # use specified tolerance to filter MVNs
#             std_cutoff = norm.ppf(1 - tol)
#             # these values define the multi-dim cuboid in which sources must lie:
#             real_cutoffs = std_cutoff * self.bandwidths
#             filt = lambda x: np.all(np.abs(self.data - x) < real_cutoffs, axis=1)
#             get_mvns = lambda x: np.array(self.mvns)[filt(x)]
#
#             it = np.nditer(args + (None,))
#             for x in it:
#                 x[self.ndim][...] = np.sum([y.pdf(*x[:-1]) for y in get_mvns(x[:-1])])
#
#             return it.operands[self.ndim] / float(self.ndata)
#
#     def values_at_data(self, **kwargs):
#         return self.pdf(*[self.data[:, i] for i in range(self.ndim)], **kwargs)
#         # return np.power(2*PI, -self.ndim * 0.5) / np.prod(self.std_devs ** 2, axis=1) / float(self.ndata)
#
#     def values_on_grid(self, n_points=10):
#         grids = np.meshgrid(*[np.linspace(mi, ma, n_points) for mi, ma in zip(self.min_array, self.max_array)])
#         it = np.nditer(grids + [None,])
#         for x in it:
#             x[-1][...] = self.pdf(*x[:-1]) # NB must unpack, as x[:-1] is a tuple
#         return it.operands[-1]
#
#
# class VariableBandwidthKdeIsotropic(FixedBandwidthKde):
#
#     def set_bandwidths(self, *args, **kwargs):
#         try:
#             self.nn = kwargs.pop('nn')
#         except KeyError:
#             # default values
#             self.nn = 10 if self.ndim == 1 else 100
#
#         # compute nn distances on unnormed data
#         kd = KDTree(self.data)
#
#         self.bandwidths = np.zeros((self.ndata, self.ndim))
#         self.mvns = []
#
#         for i in range(self.ndata):
#             d, _ = kd.query(self.data[i, :], k=self.nn)
#             nn = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]
#             # all dims have same bandwidth
#             self.bandwidths[i] = np.ones(self.ndim) * nn
#
#             self.mvns.append(MultivariateNormal(self.data[i], self.bandwidths[i]**2))
#
# class VariableBandwidthKde(FixedBandwidthKde):
#
#     def set_bandwidths(self, *args, **kwargs):
#         try:
#             self.nn = kwargs.pop('nn')
#         except KeyError:
#             # default values
#             self.nn = 10 if self.ndim == 1 else 100
#
#         # compute nn distances
#         nd = self.normed_data
#         std = self.raw_std_devs
#         kd = KDTree(nd)
#         # kd = KDTree(self.data)
#
#         self.nn_distances = np.zeros(self.ndata)
#         self.bandwidths = np.zeros((self.ndata, self.ndim))
#         self.mvns = []
#
#         for i in range(self.ndata):
#             d, _ = kd.query(nd[i, :], k=self.nn)
#             # d, _ = kd.query(self.data[i, :], k=self.nn)
#             self.nn_distances[i] = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]
#             self.bandwidths[i] = std * self.nn_distances[i]
#             # self.bandwidths[i] = np.ones(self.ndim) * self.nn_distances[i]
#
#             self.mvns.append(MultivariateNormal(self.data[i], self.bandwidths[i]**2))
#
#     @property
#     def raw_std_devs(self):
#         return np.std(self.data, axis=0)
#
#     @property
#     def normed_data(self):
#         return self.data / self.raw_std_devs


def weighted_choice_np(weights):
    totals = np.cumsum(weights)
    throw = np.random.rand()
    return np.searchsorted(totals, throw)


def weighted_choice_sub(weights):
    """ For best performance, sort weights in descending order first """
    # rnd = np.random.random() * sum(weights)
    rnd = np.random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i


def sample_events(P):
    n = P.shape[0]
    res = []
    for i in range(n):
        p = P[:, i]
        idx = np.argsort(p)
        idx = idx[::-1]
        res.append((i, idx[weighted_choice_np(p[idx])]))
    return res


def initial_guess(data):
    # TODO: something more sensible than this
    N = data.shape[0]
    # N = sim_data.shape[0]

    P = np.random.random((N, N))
    # restrict to lower triangle + 1
    P = np.triu(P, 0)
    col_sums = np.sum(P, axis=0)
    for i in range(N):
        P[:, i] /= col_sums[i]
    return P


def estimate_bg_from_sample(sample_bg):
    # use VariableBandwidthKde class to get estimates separately in time, space
    pass


# def run():
#     c = simulate.MohlerSimulation()
#     c.run()
#     data = np.array(c.data)[:, :3] # (t, x, y, b_is_BG)
#     P = initial_guess(data)
#     sample_idx = sample_events(P)
#     bg = []
#     interpoint = []
#     for x0, x1 in sample_idx:
#         if x0 == x1:
#             # bg
#             bg.append(data[x0, :])
#         else:
#             # offspring
#             dest = data[x0, :]
#             origin = data[x1, :]
#             interpoint.append(dest - origin)
#     bg_t_kde = VariableBandwidthKde(bg[:, 0])
#     bg_xy_kde = VariableBandwidthKde(bg[:, 1:])
#     trigger_kde = VariableBandwidthKde(interpoint)




