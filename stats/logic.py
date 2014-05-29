__author__ = 'gabriel'
import numpy as np
from pandas import DataFrame, Series
from django.contrib.gis.geos import Polygon, MultiPolygon, LinearRing


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


def global_morans_i(data, W):
    z = data - np.mean(data)
    try:
        # assume input is pandas datatype
        cross_m = DataFrame(np.outer(z, z), index=z.index, columns=z.index)

    except AttributeError:
        # assume input is numpy array or similar
        cross_m = np.outer(z, z)

    return (W * cross_m).values.sum() / sum(z**2)


def global_morans_i_p(data, W, n_iter=1000):
    perm_i = np.array(sorted([global_morans_i(np.random.permutation(data), W) for n in range(n_iter)]))
    real_i = global_morans_i(data, W)
    # compute p
    try:
        idx = np.where(perm_i > real_i)[0][0]
    except IndexError:
        idx = len(perm_i) - 1
    return real_i, 1-idx/float(n_iter - 1)



def local_morans_i(data, W):
    z = data - np.mean(data)
    return W.dot(z) * z


def intersection_boolean_connectivity(areal_unit_qset, distance=None):
    """ Given the input region qset, returns a normalised Boolean spatial weight matrix indicating connectivity.
        Results in QUEEN connectivity matrix. """

    n = areal_unit_qset.count()
    # W = np.zeros((n, n), dtype=float)

    all_ids = np.array([x.id for x in areal_unit_qset])
    names = [x.name for x in areal_unit_qset]
    W = DataFrame(dict([(x, 0.) for x in names]), index=names)

    for i, r in enumerate(areal_unit_qset):
        if distance:
            qset = areal_unit_qset.filter(mpoly__distance_lte=(r.mpoly, distance))
        else:
            qset = areal_unit_qset.filter(mpoly__intersects=r.mpoly)

        this_name = r.name
        # touching_ids = [x.id for x in qset if x.id != r.id]
        touching_names = [x.name for x in qset if x.id != r.id]
        for tn in touching_names:
            W[this_name][tn] = 1.
            W[tn][this_name] = 1.

    for i in range(len(W)):
        try:
            W.iloc[i] /= sum(W.iloc[i])
        except ZeroDivisionError:
            pass
    return W


def rook_boolean_connectivity(areal_unit_qset):
    """ Boolean connectivity, two regions are connected if they share a line """
    names = [x.name for x in areal_unit_qset]
    W = DataFrame(dict([(x, 0.) for x in names]), index=names)

    for r in areal_unit_qset:
        qset = areal_unit_qset.filter(mpoly__intersects=r.mpoly)
        coords = r.mpoly.coords
        for polycoords in coords:
            # discard z axis
            poly = Polygon(polycoords[0])
            for q in qset:
                if q.id == r.id:
                    # don't consider self-overlaps
                    continue
                other_coords = q.mpoly.coords
                for otherpolycoords in other_coords:
                    other_poly = Polygon(otherpolycoords[0])
                    if isinstance(poly.union(other_poly), Polygon):
                        # shared boundary
                        W[r.name][q.name] = 1.
                        W[q.name][r.name] = 1.

    for i in range(len(W)):
        try:
            W.iloc[i] /= sum(W.iloc[i])
        except ZeroDivisionError:
            pass
    return W


def create_ring(centre, radius, npts=None):
    npts = npts or min(max(10, int(2 * np.pi * radius)), 50)
    x = centre[0] + radius * np.cos(np.linspace(0, 2 * np.pi, npts + 1))
    y = centre[1] + radius * np.sin(np.linspace(0, 2 * np.pi, npts + 1))
    return LinearRing(*zip(x, y))

