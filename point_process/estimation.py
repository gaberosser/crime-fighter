__author__ = 'gabriel'
import numpy as np
import numpy
import time
import multiprocessing
PI = np.pi


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


def initial_guess(pdiff):

    N = pdiff.shape[0]
    P = np.triu(1 / (1 + pdiff[:, :, 0]), 0) / (1 + np.sqrt(pdiff[:, :, 1]**2 + pdiff[:, :, 2]**2))
    col_sums = np.sum(P, axis=0)
    for i in range(N):
        P[:, i] /= col_sums[i]
    return P


def estimate_bg_from_sample(sample_bg):
    # use VariableBandwidthKde class to get estimates separately in time, space
    pass


def _compute_threshold(x, tol):
    edf = np.arange(x.size) / float(x.size)
    idx = (edf > tol).nonzero()[0]
    if len(idx) == 0:
        return np.max(x)
    sx = np.sort(x)
    return sx[idx[0]]


def compute_trigger_thresholds(k, tol=0.99):
    """ Compute the upper thresholds on time and distance such that fraction tol of measurements are included """
    # time EDF
    t_max = k.marginal_icdf(tol, dim=0)
    x_max = k.marginal_icdf(tol, dim=1)
    y_max = k.marginal_icdf(tol, dim=2)
    d_max = (x_max + y_max) * 0.5

    return t_max, d_max


# def compute_chunk(k, d, t_max, d_max):
#
#         try:
#             return k.pdf(d[:, 0], d[:, 1], d[:, 2])
#         except Exception as e:
#             print repr(e)
#             return 0, [0]


def evaluate_trigger_kde(k, data, tol=0.95, chunksize=100000, ngrid=100):

    t_max, d_max = compute_trigger_thresholds(k, tol=tol)
    filt = lambda x: (x[:, 0] <= t_max) & (numpy.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2) <= d_max)
    i, j = np.triu_indices(data.shape[0], k=1)
    n_iter = i.size / chunksize + 1
    to_keep_idx = []

    for n in range(n_iter):
        idx = slice(n*chunksize, (n+1)*chunksize)
        this_i = i[idx]
        this_j = j[idx]
        d = data[this_j, :] - data[this_i, :]
        to_keep_idx.append(filt(d))

    to_keep_idx = np.concatenate(to_keep_idx)
    i = i[to_keep_idx]
    j = j[to_keep_idx]

    # interpolate trigger KDE
    diff_data = data[j, :] - data[i, :]
    tgrid, xgrid, ygrid = np.meshgrid(np.linspace(0, t_max, ngrid), np.linspace(-d_max, d_max, ngrid), np.linspace(-d_max, d_max, ngrid))
    f = k.pdf_interp_fn(tgrid, xgrid, ygrid)

    g = f(diff_data)
    return i, j, g
