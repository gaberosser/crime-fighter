__author__ = 'gabriel'
import numpy as np
import simulate
from scipy.spatial import KDTree
from scipy.stats import norm
import operator
PI = np.pi


class MultivariateNormal():

    def __init__(self, mean, vars):
        self.ndim = len(vars)
        self.norms = [lambda x: self.norm1d(x, m, v) for (m, v) in zip(mean, vars)]

    def pdf(self, x):
        if x.shape[-1] != self.ndim:
            raise AttributeError("Incorrect dimensions for input variable")
        return np.prod([t(x) for t in self.norms])

    def norm1d(self, x, mu, var):
        return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))


class VariableBandwidthKde():

    def __init__(self, data, nn=None):
        self.data = data
        if len(data.shape) == 1:
            self.data = np.array(data).reshape((len(data), 1))

        if nn:
            self.nn = nn
        else:
            # default values
            nn = 10 if self.ndim == 1 else 100

        # compute nn distances
        nd = self.normed_data
        std = self.raw_std_devs
        kd = KDTree(nd)

        self.nn_distances = np.zeros(self.ndata)
        self.std_devs = np.zeros((self.ndata, self.ndim))
        self.mvns = []

        for i in range(self.ndata):
            d, _ = kd.query(nd[i, :], k=nn)
            self.nn_distances[i] = max(d[~np.isinf(d)]) if np.isinf(d[-1]) else d[-1]
            self.std_devs[i] = std * self.nn_distances[i]

            self.mvns.append(MultivariateNormal(self.data[i], self.std_devs[i]**2))

    def pdf(self, datum):
        ## TODO: support vector inputs?
        return sum([x.pdf(datum) for x in self.mvns]) / float(self.ndata)

    @property
    def values_at_data(self):
        return np.power(2*PI, -self.ndim * 0.5) * np.prod(self.std_devs, axis=1) / float(self.ndata)

    @property
    def ndim(self):
        return self.data.shape[1]

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def raw_std_devs(self):
        return np.std(self.data, axis=0)

    @property
    def normed_data(self):
        return self.data / self.raw_std_devs


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


def run():
    c = simulate.MohlerSimulation()
    c.run()
    data = np.array(c.data)[:, :3]
    P = initial_guess(data)
    sample_idx = sample_events(P)
    bg = []
    interpoint = []
    for x0, x1 in sample_idx:
        if x0 == x1:
            # bg
            bg.append(data[x0, :])
        else:
            # offspring
            dest = data[x0, :]
            origin = data[x1, :]
            interpoint.append(dest - origin)




