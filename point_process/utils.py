__author__ = 'gabriel'
import numpy as np


def pairwise_differences_indices(n):

    dtypes = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]

    dtype = None
    # find appropriate datatype
    for d in dtypes:
        if np.iinfo(d).max >= (n - 1):
            dtype = d
            break

    if not dtype:
        raise MemoryError("Unable to index an array this large.")

    idx_i = np.zeros(n* (n - 1) / 2, dtype=dtype)
    idx_j = np.zeros_like(idx_i)

    tally = 0
    for i in range(n):
        idx_i[tally:(tally + n - i - 1)] = np.ones(n - i - 1, dtype=dtype) * i
        idx_j[tally:(tally + n - i - 1)] = np.arange(i + 1, n, dtype=dtype)
        tally += n - i - 1

    return idx_i, idx_j


def euclidean_distance(loc1, loc2):
    # loc1 and loc2 are (N x 2) arrays.  Returns the distance between each of the N pairs of datapoints
    # return np.sqrt((loc2[:, 0] - loc1[:, 0])**2 + (loc2[:, 1] - loc1[:, 1])**2)
    return np.sqrt(np.sum((loc2 - loc1) ** 2, axis=1))


def network_distance(loc1, loc2):
    raise NotImplementedError()


def linkages_euclidean_array(data_source, max_t, max_d, data_target=None, chunksize=2**16):
    """
    Compute the indices of datapoints that are within the following tolerances:
    interpoint distance less than max_d
    time difference greater than zero, less than max_t
    The sign convention is (target - source).  Distances are euclidean.
    :param data_source: (N x d) array of source data, where N is number of datapoints and d is the dimensionality.
    The first dimension is always time.
    :param max_t: maximum time difference (minimum is always zero)
    :param max_d: maximum spatial distance
    :param data_target: optional (N x d) array of data.  If supplied, the linkage indices are between data_source and
    data_target, otherwise the two are set equal
    :param chunksize: The size of an iteration chunk.
    :return: tuple (idx_array_source, idx_array_target),
    """

    ndata_source = data_source.shape[0]
    if data_target is not None:
        ndata_target = data_target.shape[0]
        chunksize = min(chunksize, ndata_source * ndata_target)
        idx_i, idx_j = np.meshgrid(range(ndata_source), range(ndata_target), copy=False)
    else:
        # self-linkage
        data_target = data_source
        chunksize = min(chunksize, ndata_source * (ndata_source - 1) / 2)
        idx_i, idx_j = pairwise_differences_indices(ndata_source)

    link_i = []
    link_j = []
    for k in range(0, idx_i.size, chunksize):
        i = idx_i.flat[k:(k + chunksize)]
        j = idx_j.flat[k:(k + chunksize)]
        dt = data_target[j, 0] - data_source[i, 0]
        dd = euclidean_distance(data_target[j, 1:], data_source[i, 1:])
        mask = (dt <= max_t) & (dt > 0.) & (dd <= max_d)
        link_i.extend(i[mask])
        link_j.extend(j[mask])

    return np.array(link_i), np.array(link_j)


def linkages(data_source, max_t, max_d, data_target=None, chunksize=2**16):
    """
    Compute the indices of datapoints that are within the following tolerances:
    interpoint distance less than max_d
    time difference greater than zero, less than max_t
    The sign convention is (target - source).  Distances are euclidean.
    :param data_source: EuclideanSpaceTimeData array of source data.  Must be sorted by time ascending.
    :param max_t: maximum time difference (minimum is always zero)
    :param max_d: maximum spatial distance
    :param data_target: optional EuclideanSpaceTimeData array.  If supplied, the linkage indices are between
    data_source and data_target, otherwise the two are set equal
    :param chunksize: The size of an iteration chunk.
    :return: tuple (idx_array_source, idx_array_target),
    """
    ndata_source = data_source.ndata
    if data_target is not None:
        ndata_target = data_target.ndata
        chunksize = min(chunksize, ndata_source * ndata_target)
        idx_i, idx_j = np.meshgrid(range(ndata_source), range(ndata_target), copy=False)
    else:
        # self-linkage
        data_target = data_source
        chunksize = min(chunksize, ndata_source * (ndata_source - 1) / 2)
        idx_i, idx_j = pairwise_differences_indices(ndata_source)

    link_i = []
    link_j = []
    for k in range(0, idx_i.size, chunksize):
        i = idx_i.flat[k:(k + chunksize)]
        j = idx_j.flat[k:(k + chunksize)]
        dt = (data_target.time[j] - data_source.time[i]).flat
        dd = (data_target.space[j].distance(data_source.space[i])).flat
        mask = (dt <= max_t) & (dt > 0.) & (dd <= max_d)
        link_i.extend(i[mask])
        link_j.extend(j[mask])

    return np.array(link_i), np.array(link_j)


# def _set_linkages_iterated(self, data=None, chunksize=2**16):
#     """ Iteration-based approach to computing parent-offspring couplings, required when memory is limited """
#
#     data = data if data is not None else self.data
#     ndata = data.shape[0]
#
#     chunksize = min(chunksize, ndata * (ndata - 1) / 2)
#     idx_i, idx_j = pairwise_differences_indices(ndata)
#     link_i = []
#     link_j = []
#
#     for k in xrange(0, len(idx_i), chunksize):
#         i = idx_i[k:(k + chunksize)]
#         j = idx_j[k:(k + chunksize)]
#         t = data[j, 0] - data[i, 0]
#         d = np.sqrt((data[j, 1] - data[i, 1])**2 + (data[j, 2] - data[i, 2])**2)
#         mask = (t <= self.max_trigger_t) & (d <= self.max_trigger_d)
#         link_i.extend(i[mask])
#         link_j.extend(j[mask])
#
#     return (np.array(link_i), np.array(link_j))
#
#
#
# def _target_source_linkages(self, t, x, y, data=None, chunksize=2**16):
#     """ Iteration-based approach to computing parent-offspring couplings for the arbitrary target locations
#         supplied in t, x, y arrays.  Optionally can supply different data, in which case this is treated as the
#         putative parent data instead of self.data.
#         :return: Tuple of two index arrays, with same interpretation as self.linkages
#         NB indices are treated as if t, x, y are flat"""
#
#     data = data if data is not None else self.data
#     ndata = data.shape[0]
#
#     chunksize = min(chunksize, ndata ** 2)
#     idx_i, idx_j = np.meshgrid(range(ndata), range(t.size), copy=False)
#     link_i = []
#     link_j = []
#
#     for k in range(0, idx_i.size, chunksize):
#         i = idx_i.flat[k:(k + chunksize)]
#         j = idx_j.flat[k:(k + chunksize)]
#         tt = t.flat[j] - data[i, 0]
#         dd = np.sqrt((x.flat[j] - data[i, 1])**2 + (y.flat[j] - data[i, 2])**2)
#         mask = (tt <= self.max_trigger_t) & (tt > 0.) & (dd <= self.max_trigger_d)
#         link_i.extend(i[mask])
#         link_j.extend(j[mask])
#
#     return np.array(link_i), np.array(link_j)