__author__ = 'gabriel'

import cProfile
import pstats
import models
import kde
import numpy as np
import os
PROFILE_DIR = kde.__path__[0]


def run_vkde3(num_sources=1000):
    num_dim = 3
    n = round(np.power(num_sources, 1/float(num_dim)))
    if n**num_dim != num_sources:
        raise Exception('Must specify a number of sources with an integer sqrt.')
    source_loc = np.meshgrid(*([np.linspace(0, 1, n)]*num_dim))
    sources = np.vstack([x.flatten() for x in source_loc]).transpose()
    k = models.VariableBandwidthKde(sources, nn=2)
    return k, k.values_at_data()


def profile_vkde_3d(num_sources=1000):
    prof_file = os.path.join(PROFILE_DIR, 'kde_3d.prof')
    z = cProfile.runctx('run_vkde3(num_sources)', {'num_sources': num_sources, 'run_vkde3': run_vkde3}, {}, filename=prof_file)
    p = pstats.Stats(prof_file)
    p.strip_dirs()
    p.sort_stats('cumulative').print_stats(20)
    return z
