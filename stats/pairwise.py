__author__ = 'gabriel'
import numpy as np
from scipy import stats
import warnings


WILCOX_LOOKUP = {
    0.05: {
        6: 0,
        7: 2,
        8: 4,
        9: 6,
        10: 8,
        11: 11,
        12: 14,
        13: 17,
        14: 21,
        15: 25,
        16: 30,
        17: 35,
        18: 40,
        19: 46,
        20: 52,
        21: 59,
        22: 66,
        23: 73,
        24: 81,
        25: 89,
    },
    0.025: {
        7: 0,
        8: 2,
        9: 3,
        10: 5,
        11: 7,
        12: 10,
        13: 13,
        14: 16,
        15: 20,
        16: 24,
        17: 28,
        18: 33,
        19: 38,
        20: 43,
        21: 49,
        22: 56,
        23: 62,
        24: 69,
        25: 77,
    },
    0.01: {
        8: 0,
        9: 2,
        10: 3,
        11: 5,
        12: 7,
        13: 10,
        14: 13,
        15: 16,
        16: 20,
        17: 23,
        18: 28,
        19: 32,
        20: 38,
        21: 43,
        22: 49,
        23: 55,
        24: 61,
        25: 68,
    }
}


def wilcoxon_one_tailed(v2, v1):
    """
    Wilcoxon signed rank test, ONE TAILED.
    Null hypothesis is median difference is zero.
    Alternative hypothesis: median(v2) > median(v1) NB ALWAYS IN THIS ORDER
    :param v2:
    :param v1:
    :param one_tailed:
    :return:
    """

    # perform a two-tailed test
    W, p = wilcoxon(v2, v1)
    p *= 0.5

    d = v2 - v1
    # manually check the one-sided assumption - this is wrong by definition if the medians are the other way around
    # in such a case report p = 0.5, meaning p >= 0.5

    if np.nanmedian(d) < 0:
        return W, 0.5

    return W, p




    # s = np.sign(v2 - v1)
    # d = v2 - v1
    #
    # # Wilcox correction: ignore zeros
    # d = d[d != 0.]
    #
    # # ranking: NB, matches are given the AVERAGE RANK
    # _, _, this_rank, this_count = np.unique(np.abs(d), True, True, True)
    # this_rank += 1
    #
    # m = this_rank.max()
    # i = 1
    # while i <= d.size:
    #
    #     n = sum(this_rank == i)
    #     if n > 1:
    #         # new rank is the MEAN of the tied ranks
    #         new_rank = np.arange(i, i + n).mean()
    #         this_rank[this_rank > i] += n - 1  # increment all further ranks
    #         this_rank[this_rank == i] = new_rank  # replace all matching ranks
    #     i += n
    #
    #
    # Wplus = np.nansum(this_rank[d > 0])
    # Wminus = np.nansum(this_rank[d < 0])
    # W = min(Wplus, Wminus)
    # n = float(d.size)
    #
    # # null hypothesis with normal asssumption -> Z value -> p-value
    # if n >= 10:
    #     mn = n*(n + 1.) * 0.25
    #     se = np.sqrt(n * (n + 1.) * (2. * n + 1.) / 24.)
    #     z = (W - mn) / se
    #
    # prob = stats.distributions.norm.sf(abs(z))
    # if not one_tailed:
    #     prob *= 2
    #
    #
    # Z = (4 * W - n * (n + 1)) / (np.sqrt(2 * n * (n + 1) * (2 * n + 1) / 3) * (p_plus + p_minus - (p_plus - p_minus) ** 2))
    #
    # return Z, stats.norm.sf(np.abs(Z))


def wilcoxon(x, y=None, zero_method="wilcox", correction=False):
    """
    Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but spliting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.

    Returns
    -------
    T : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    p-value : float
        The two-sided p-value for the test.

    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    """

    if not zero_method in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' \
                          or 'pratt' or 'zsplit'")

    if y is None:
        d = x
    else:
        x, y = map(np.asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x-y

    if zero_method == "wilcox":
        d = np.compress(np.not_equal(d, 0), d, axis=-1)  # Keep all non-zero differences

    count = len(d)
    if (count < 10):
        warnings.warn("Warning: sample size too small for normal approximation.")
    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = min(r_plus, r_minus)
    mn = count*(count + 1.) * 0.25
    se = count*(count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * stats.distributions.norm.sf(abs(z))
    return T, prob
