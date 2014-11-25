__author__ = 'gabriel'
import numpy as np
from scipy import sparse


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
        dt = (data_target.time.getrows(j) - data_source.time.getrows(i)).toarray(0)
        dd = (data_target.space.getrows(j).distance(data_source.space.getrows(i))).toarray(0)
        mask = (dt <= max_t) & (dt > 0.) & (dd <= max_d)
        link_i.extend(i[mask])
        link_j.extend(j[mask])

    return np.array(link_i), np.array(link_j)


def augmented_matrix(new_p, old_p):
    """
    Create an augmented matrix based on the previous version, but with new datapoints added
    This assumes the new dataset builds on the previous one
    """

    num_old = old_p.shape[0]
    num_new = new_p.shape[0]

    assert num_new > num_old

    # combine old and new indices
    pre_indices = old_p.indices
    pre_indptr = old_p.indptr
    new_indices = new_p.indices
    new_indptr = new_p.indptr

    comb_indices = np.concatenate((pre_indices, new_indices[new_indptr[num_old]:]))
    comb_indptr = np.concatenate((pre_indptr[:-1], new_indptr[num_old:]))
    comb_data = np.concatenate((old_p.data, new_p.data[old_p.nnz:]))
    comb_p = sparse.csc_matrix((comb_data, comb_indices, comb_indptr), shape=(num_new, num_new)).tocsr()

    return comb_p