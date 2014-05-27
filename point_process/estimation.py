__author__ = 'gabriel'
import numpy as np
import operator
import time
import multiprocessing
PI = np.pi


def weighted_choice_np(weights):
    totals = np.cumsum(weights)
    prng = np.random.RandomState()
    throw = prng.rand()
    return np.searchsorted(totals, throw)


def sample_events(P):
    n = P.shape[0]
    res = []
    for i in range(n):
        p = P[:, i]
        idx = np.argsort(p)
        idx = idx[::-1]
        res.append((i, idx[weighted_choice_np(p[idx])]))
    return res


def sample_bg_and_interpoint(data, P):
    if data.shape[0] != P.shape[0]:
        raise AttributeError("Dimensions of data and P do not match")

    sample_idx = sample_events(P)
    bg = []
    interpoint = []
    for effect, cause in sample_idx:
        if cause == effect:
            # bg
            bg.append(data[cause, :])
        else:
            # offspring
            dest = data[effect, :]
            origin = data[cause, :]
            interpoint.append(dest - origin)

    return np.array(bg), np.array(interpoint)


def pairwise_differences(data, b_iter=False, dtype=None):
    """ Compute pairwise (t, x, y) difference vector.
        Setting b_iter=True calls an alternative iteration-based calculation to save memory. """

    dtype = dtype or np.float64

    if b_iter:
        # use this routine if the system memory is limited:
        ndata = data.shape[0]
        pdiff = np.zeros((ndata, ndata, 3))
        for i in range(ndata):
            for j in range(i):
                pdiff[j, i, :] = data[i] - data[j]
        return pdiff

    td = reduce(operator.sub, np.meshgrid(data[:, 0], data[:, 0], copy=False))
    xd = reduce(operator.sub, np.meshgrid(data[:, 1], data[:, 1], copy=False))
    yd = reduce(operator.sub, np.meshgrid(data[:, 2], data[:, 2], copy=False))
    return np.dstack((td, xd, yd)).astype(dtype)


def initial_guess(data):

    N = data.shape[0]
    pdiff = pairwise_differences(data)
    c = 5
    P = np.triu(1 / (1 + c * pdiff[:, :, 0]), 0) / (1 + c * np.sqrt(pdiff[:, :, 1]**2 + pdiff[:, :, 2]**2))
    col_sums = np.sum(P, axis=0)
    for i in range(1, N):
        P[i, i] = (col_sums[i] - 1.)
        P[:, i] /= (2 * (col_sums[i] - 1))
    return P


def initial_guess_educated(data, ct=None, cd=None):

    N = data.shape[0]
    pdiff = pairwise_differences(data)
    ct = ct or 1
    cd = cd or 10
    dt = 1 / (1 + ct * pdiff[:, :, 0])
    dd = 1 / (1 + cd * np.sqrt(pdiff[:, :, 1] ** 2 + pdiff[:, :, 2] ** 2))
    P = np.triu(dt * dd, 0)
    col_sums = np.sum(P, axis=0)
    P /= col_sums
    return P


def initial_guess_equality(data):

    N = data.shape[0]
    P = np.ones((N, N))
    P = np.triu(P)
    # P = np.eye(N) * np.arange(N)
    # P[0, 0] = 1
    # P[np.triu_indices(N, k=1)] = 1.
    col_sums = np.sum(P, axis=0)
    P /= col_sums
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

    return t_max, x_max, y_max


def find_acceptance_region_interpoints(k, data, tol=0.95):
    """ Find all interpoint vectors that fall within an acceptance region (cuboid) specified by tol. """

    t_max = k.marginal_icdf(tol, dim=0)
    x_max = k.marginal_icdf(tol, dim=1)
    y_max = k.marginal_icdf(tol, dim=2)

    filt = lambda x: (x[:, 0] > 0) & (x[:, 0] <= t_max) \
        & (x[:, 1] >= -x_max) & (x[:, 1] <= x_max) \
        & (x[:, 2] >= -y_max) & (x[:, 2] <= y_max)
    i, j = np.triu_indices(data.shape[0], k=1)

    to_keep_idx = filt(data[j, :] - data[i, :])
    i = i[to_keep_idx]
    j = j[to_keep_idx]

    # interpolate trigger KDE
    diff_data = data[j, :] - data[i, :]
    return i, j, diff_data

# def evaluate_trigger_kde(k, data, tol=0.95, ngrid=100):
#     """ Approximate method to evaluate the trigger KDE, designed to work around the very large number of targets
#         typically requested.
#         First filter by acceptance region, based on tolerance.
#         Next, generate a 3D grid of points and use zero-order (nn) interpolation to approximate all data within
#         acceptance region.
#     """
#     i, j, diff_data = find_acceptance_region_interpoints(k, data, tol=tol)
#     t_max = np.max(diff_data[:, 0])
#     x_min = np.min(diff_data[:, 1])
#     x_max = np.max(diff_data[:, 1])
#     y_min = np.min(diff_data[:, 2])
#     y_max = np.max(diff_data[:, 2])
#     tgrid, xgrid, ygrid = np.meshgrid(np.linspace(0, t_max, ngrid), np.linspace(x_min, x_max, ngrid), np.linspace(y_min, y_max, ngrid))
#     f = k.pdf_interp_fn(tgrid, xgrid, ygrid)
#
#     ndata = data.shape[0]
#     g = np.zeros((ndata, ndata))
#     g[i, j] = f(diff_data)
#     return g


def evaluate_trigger_kde(k, data, linkage):
    """ Evaluate trigger KDE at the interpoint distances given by the indices in linkage arrays """
    ndata = data.shape[0]
    diff_data = data[linkage[1], :] - data[linkage[0], :]
    g = np.zeros((ndata, ndata))
    g[linkage] = k.pdf(diff_data[:, 0], diff_data[:, 1], diff_data[:, 2]) / float(ndata)
    return g