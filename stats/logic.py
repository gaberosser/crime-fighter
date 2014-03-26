__author__ = 'gabriel'
import numpy as np

def truncated_acf_1d(x, max_lag=None):
    t = len(x)
    if max_lag:
        if max_lag > t:
            raise Exception('specified max_lag is too large for the supplied data')
    else:
        max_lag = t/4 # NB int
    v = np.var(x)
    z = x - np.mean(x) # centered series
    res = []
    for k in range(max_lag):
        idx0 = np.arange(t - k)
        idx1 = np.arange(k, t)
        x0 = z[idx0]
        x1 = z[idx1]
        res.append(np.dot(x0, x1)/float(t - k))

    return np.array(res)/v


def conv_acf_1d(x, max_lag=None):
    t = len(x)
    if max_lag:
        if max_lag > t:
            raise Exception('specified max_lag is too large for the supplied data')
    else:
        max_lag = t/4 # NB int

    v = np.var(x)
    z = x - np.mean(x) # centered

    tmp = np.convolve(z, z[::-1], mode='full')[-t:] / v
    tmp = tmp[:max_lag]
    return tmp / np.arange(t, t - max_lag, -1)


def yw_pacf_1d(acf):
    p = len(acf) - 1
    r = acf[1:] # r_1, ..., r_{p}
    y = r[:-1] # r_1, ..., r_{p-1}
    x = np.concatenate((y[::-1], [1], y)) # r_{p-1}, ..., r_1, 1, r_1, ..., r_{p-1}

    idx_fun = lambda n: range(p - 1 - n, 2*p - 1 - n)

    R = np.array([x[idx_fun(n)] for n in range(p)]) # R^(p)
    return np.array([1.] + [np.linalg.solve(R[:i+1,:i+1], r[:i+1])[-1] for i in range(p)])


def boolean_connectivity(areal_unit_qset):
    """ Given the input region qset, returns a normalised Boolean spatial weight matrix indicating connectivity """
    n = areal_unit_qset.count()
    W = np.zeros((n, n), dtype=float)
    all_ids = np.array([x.id for x in areal_unit_qset])
    for i, r in enumerate(areal_unit_qset):
        touching_ids = [x.id for x in areal_unit_qset.filter(mpoly__intersects=r.mpoly) if x.id != r.id]
        # get idx
        idx = np.where(np.in1d(all_ids, touching_ids))[0]
        if len(idx):
            try:
                W[i, idx] = 1.
                W[idx, i] = 1.
            except Exception:
                import pdb; pdb.set_trace()
    for i in range(len(W)):
        W[i] = W[i]/sum(W[i])
    return W



