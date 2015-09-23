__author__ = 'gabriel'
import numpy as np
from scipy import stats


def wilcoxon_signed_rank(v2, v1):

    d = v2 - v1
    s = np.sign(d)
    d = np.abs(d)

    # ranking: NB, matches are given the AVERAGE RANK
    _, _, this_rank, this_count = np.unique(d, True, True, True)
    this_rank += 1

    m = this_rank.max()
    i = 1
    while i <= d.size:

        n = sum(this_rank == i)
        if n > 1:
            # new rank is the MEAN of the tied ranks
            new_rank = np.arange(i, i + n).mean()
            this_rank[this_rank > i] += n - 1  # increment all further ranks
            this_rank[this_rank == i] = new_rank  # replace all matching ranks
        i += n

    W = np.nansum(s * this_rank)
    n = float(d.size)
    p_plus = np.sum(d > 0) / n
    p_minus = np.sum(d < 0) / n


    Z = (4 * W - n * (n + 1)) / (np.sqrt(2 * n * (n + 1) * (2 * n + 1) / 3) * (p_plus + p_minus - (p_plus - p_minus) ** 2))

    return Z, stats.norm.sf(np.abs(Z))

