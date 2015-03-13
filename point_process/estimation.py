__author__ = 'gabriel'
import numpy as np
import operator
import gc
from scipy import sparse
PI = np.pi

## FIXME: global prng probably isn't the best idea
prng = np.random.RandomState()


def set_seed(seed=None):
    prng.seed(seed)


def weighted_choice_np(weights):
    totals = np.cumsum(weights)
    throw = prng.rand()
    return np.searchsorted(totals, throw)


# def sample_events(P):
#     n = P.shape[1]
#     res = []
#     for i in range(n):
#         p = P[:, i]
#         idx = np.argsort(p)
#         idx = idx[::-1]
#         res.append((i, idx[weighted_choice_np(p[idx])])) # effect, cause
#     return res
#
#
# def sample_bg_and_interpoint(data, P):
#     if data.shape[0] != P.shape[0]:
#         raise AttributeError("Dimensions of data and P do not match")
#
#     sample_idx = sample_events(P)
#     bg = []
#     interpoint = []
#     cause_effect = []
#     for effect, cause in sample_idx:
#         if cause == effect:
#             # bg
#             bg.append(data[cause, :])
#         else:
#             # offspring
#             dest = data[effect, :]
#             origin = data[cause, :]
#             interpoint.append(dest - origin)
#             cause_effect.append((cause, effect))
#
#     return np.array(bg), np.array(interpoint), np.array(cause_effect)


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


def estimator_bowers(data, linkage, ct=1, cd=10, matrix_init=sparse.csr_matrix):

    n = data.shape[0]
    P = sparse.lil_matrix((n, n))

    # off-diagonal

    tt = 1 / (1 + ct * (data[linkage[1], 0] - data[linkage[0], 0]))
    dd = 1 / (1 + cd * np.sqrt(
        (data[linkage[1], 1] - data[linkage[0], 1]) ** 2 +
        (data[linkage[1], 2] - data[linkage[0], 2]) ** 2
    ))

    diag_linkage = (np.arange(n), np.arange(n))

    P_trig = sparse.csr_matrix((tt * dd, linkage), shape=(n, n))
    P_bg = sparse.csr_matrix((np.ones(n), diag_linkage), shape=(n, n))
    P = P_trig + P_bg
    colsums = P.sum(axis=0).flat
    P_trig[linkage] = P_trig[linkage] / colsums[linkage[1]]
    P_bg[diag_linkage] = P_bg[diag_linkage] / colsums[diag_linkage[1]]

    P = P_trig + P_bg
    return matrix_init(P)


def estimator_exp_gaussian(data, linkage, ct, cd):
    n = data.shape[0]

    # off-diagonal

    tt = ct * np.exp(-ct * (data[linkage[1], 0] - data[linkage[0], 0]))
    dd_k = np.sqrt(2 / (np.pi * cd))
    dd_sq = (data[linkage[1], 1] - data[linkage[0], 1]) ** 2 + (data[linkage[1], 2] - data[linkage[0], 2]) ** 2
    dd = dd_k * np.exp(-dd_sq / (2 * cd ** 2))

    diag_linkage = (np.arange(n), np.arange(n))

    P_trig = sparse.csr_matrix((tt * dd, linkage), shape=(n, n))
    P_bg = sparse.csr_matrix((np.ones(n), diag_linkage), shape=(n, n))
    P = P_trig + P_bg
    colsums = P.sum(axis=0).flat
    P_trig[linkage] = P_trig[linkage] / colsums[linkage[1]]
    P_bg[diag_linkage] = P_bg[diag_linkage] / colsums[diag_linkage[1]]

    P = P_trig + P_bg
    return P


def initial_guess_educated(data, ct=None, cd=None):

    ct = ct or 1
    cd = cd or 10

    # pdiff = pairwise_differences(data)

    td = reduce(operator.sub, np.meshgrid(data[:, 0], data[:, 0], copy=False))
    xd = reduce(operator.sub, np.meshgrid(data[:, 1], data[:, 1], copy=False))
    yd = reduce(operator.sub, np.meshgrid(data[:, 2], data[:, 2], copy=False))

    dt = 1 / (1 + ct * td)
    del td
    dd = 1 / (1 + cd * np.sqrt(xd ** 2 + yd ** 2))
    del xd, yd
    gc.collect()

    # dt = 1 / (1 + ct * pdiff[:, :, 0])
    # dd = 1 / (1 + cd * np.sqrt(pdiff[:, :, 1] ** 2 + pdiff[:, :, 2] ** 2))
    idx = np.tril_indices_from(dt, k=-1)
    P = dt * dd
    P[idx] = 0.
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
