__author__ = 'gabriel'
import numpy as np
from scipy import sparse
import logging
import multiprocessing as mp
from data.models import DataArray

logger = logging.getLogger(__name__)

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


def linkages(data_source, threshold_fun, data_target=None, chunksize=2**18, remove_coincident_pairs=False):
    """
    Compute the indices of datapoints that are within the following tolerances:
    interpoint distance less than max_d
    time difference greater than zero, less than max_t
    The sign convention is (target - source).  Distances are euclidean.
    :param data_source: EuclideanSpaceTimeData array of source data.  Must be sorted by time ascending.
    :param threshold_fun: function that operates on the DataArray (delta_t, delta_d) returning True for linked points
    NB: also require that delta_t > 0.
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
        dt = (data_target.time.getrows(j) - data_source.time.getrows(i))
        dd = (data_target.space.getrows(j).distance(data_source.space.getrows(i)))
        mask = threshold_fun(dt, dd) & (dt > 0)
        if remove_coincident_pairs:
            mask &= (dd != 0)
        mask = mask.toarray(0)
        link_i.extend(i[mask])
        link_j.extend(j[mask])

    return np.array(link_i), np.array(link_j)


def linkage_func_separable(max_t, max_d):
    def func(dt, dd):
        return (dt <= max_t) & (dd <= max_d)
    return func


def linkage_mask(data_source, data_target, max_t, max_d, remove_coincident_pairs):
    dt = (data_target.time - data_source.time).toarray(0)
    dd = (data_target.space.distance(data_source.space)).toarray(0)
    mask = (dt <= max_t) & (dt > 0.) & (dd <= max_d)
    if remove_coincident_pairs:
        mask = mask & (dd != 0)
    return mask


def compute_linkage_chunk(this_idx, idx_i, idx_j, data_source, data_target, max_t, max_d, remove_coincident_pairs):
    i = idx_i.flat[this_idx]
    j = idx_j.flat[this_idx]
    mask = linkage_mask(data_source.getrows(i), data_target.getrows(j), max_t, max_d, remove_coincident_pairs)
    return i[mask], j[mask]


def parallel_linkages(data_source, max_t, max_d, data_target=None, chunksize=2**18, remove_coincident_pairs=False):

    ## FIXME: something here won't pickle. Try using pathos. Have to get it installed first though!!

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

    pool = mp.Pool()

    processes = []
    for k in range(0, idx_i.size, chunksize):
        this_idx = range(k, k + chunksize)
        processes.append(
            pool.apply_async(compute_linkage_chunk, args=(
                this_idx,
                idx_i,
                idx_j,
                data_source,
                data_target,
                max_t,
                max_d,
                remove_coincident_pairs,
            ))
        )

    link_i = []
    link_j = []

    for p in processes:
        ii, jj = p.get()
        link_i.append(ii)
        link_j.append(jj)

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


def random_sample_from_p(p, linkage_cols, rng=None):
    """
    Weighted sampling algorithm by Efraimidis and Spirakis. Weighted random sampling with a reservoir.
    Information Processing Letters 97 (2006) 181-185
    """
    ndata = p.shape[0]
    rng = rng or np.random.RandomState()

    ## TODO: should be able to reform linkage_cols fairly easily from p if necessary?
    # if linkage_cols is None:
    #     linkage_cols = {}
    #     for i in range(ndata):
    #         idx = p[:i+1, i].nonzero()[0]
    #         linkage_cols[i] = idx

    urvs = rng.rand(p.nnz)
    ks_matrix = p.copy()
    ks_matrix.data = np.power(urvs, 1. / p.data)


    # find the largest value in each column
    causes = [linkage_cols[n][np.argmax(ks_matrix[:, n].data)] for n in range(ndata)]
    effects = range(ndata)

    bg_idx = [x for x, y in zip(causes, effects) if x == y]

    if not len(bg_idx):
        logger.warn("No BG events remaining")

    cause_effect = zip(*[(x, y) for x, y in zip(causes, effects) if x != y])
    if not len(cause_effect):
        logger.warn("No trigger events remaining")
        cause_idx = []
        effect_idx = []
    else:
        cause_idx, effect_idx = cause_effect

    return bg_idx, list(cause_idx), list(effect_idx)


def all_bg_log_likelihood(sepp_obj):
    """
    Compute the log likelihood associated with a BG only model.
    :param sepp_obj: An instance that inherits from Sepp
    :return: Log likelihood
    """
    data = sepp_obj.data
    bg_class = sepp_obj.bg_kde_class
    bg_kde = bg_class(data, **sepp_obj.bg_kde_kwargs)
    m = bg_kde.pdf(data, normed=False)
    return np.log(m).sum()


def all_trig_log_likelihood(sepp_obj, parallel=True):
    """
    Compute the log likelihood associated with a trigger only model.
    :param sepp_obj: An instance that inherits from Sepp
    :param parallel: This computation can be quite slow, so set True if possible.
    :return: Log likelihood
    """
    interpoint_data = sepp_obj.interpoint_data
    trig_class = sepp_obj.trigger_kde_class
    ## FIXME: this should be a weighted KDE, or sampling is needed to determine a single parent-offspring relationship
    trig_kde = trig_class(interpoint_data, **sepp_obj.trigger_kde_kwargs)
    trigger = trig_kde.pdf(interpoint_data, normed=False) / sepp_obj.ndata
    g = sparse.csr_matrix((trigger, sepp_obj.linkage), shape=(sepp_obj.ndata, sepp_obj.ndata))
    # sum_g = np.array(g.sum(axis=0)).flatten()[sepp_obj.linkage[1]]
    sum_g = np.array(g.sum(axis=0)).flatten()
    return np.log(sum_g).sum()