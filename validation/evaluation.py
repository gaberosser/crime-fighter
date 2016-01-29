import numpy as np
from bisect import bisect_left
from stats.pairwise import wilcoxon


def wilcoxon_comparison(coverage_arr1, 
                        hit_rate_arr1, 
                        coverage_arr2,
                        hit_rate_arr2,
                        n_coverage_pts=200,
                        max_coverage=1.):
    """ For a range of coverage levels between 0 and 1, carry out a Wilcoxon
    signed rank test of methods 1 and 2. This is done precisely, with the cutoff
    index being recomputed for every day (rather than using the mean). 
    :param coverage_arr1: M x N matrix, where M is the number of prediction days
    and N is the number of coverage levels tested (i.e. the number of validation
    sample points).
    :param hit_rate_arr1: M x N matrix of hit rate values.
    :return: 
        covs: array containing the coverage levels
        pvals: array of length n_coverage_pts
        effect_direction: array of length n_coverage_pts, +1 means arr1 > arr2
        mean_delta: array of length n_coverage_pts, arr1 - arr2 by definition
    """
    
    covs = np.linspace(0, max_coverage, n_coverage_pts)
    m1, n1 = coverage_arr1.shape
    m2, n2 = coverage_arr1.shape
    assert m1 == m2, "Number of days must be the same for each run"
    assert coverage_arr1.shape == hit_rate_arr1.shape, "Arrays 1 must have matching shapes."
    assert coverage_arr2.shape == hit_rate_arr2.shape, "Arrays 1 must have matching shapes."
    pvals = np.zeros(n_coverage_pts)
    effect_direction = np.zeros(n_coverage_pts)
    mean_delta = np.zeros(n_coverage_pts)
    for k, c in enumerate(covs):
        idx1 = [bisect_left(coverage_arr1[i, :], c) for i in range(m1)]
        idx2 = [bisect_left(coverage_arr2[i, :], c) for i in range(m2)]
        y1 = hit_rate_arr1[range(m1), idx1]
        y2 = hit_rate_arr2[range(m2), idx2]
        _, p, e = wilcoxon(y1, y2)
        pvals[k] = p
        effect_direction[k] = e
        mean_delta[k] = y1.mean() - y2.mean()
        
    return covs, pvals, effect_direction, mean_delta