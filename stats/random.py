import numpy as np


def weighted_random_selection(weights, n=1, prng=None):
    """
    Select the *indices* of n points at random, weighted by the supplied weights matrix
    :param weights:
    :param n:
    :param prng: Optionally supply a np.random.Random() instance if control is required
    :return: Iterable of n indices, referencing the weights array, or scalar index if n == 1
    """
    prng = prng or np.random.RandomState()
    totals = np.cumsum(weights)
    throws = prng.rand(n) * totals[-1]
    res = np.searchsorted(totals, throws)
    if n == 1:
        return res[0]
    else:
        return res
