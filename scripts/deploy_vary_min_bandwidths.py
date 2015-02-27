__author__ = 'gabriel'

import vary_min_bandwidths
import numpy as np


min_t_bds = [0, 0.5, 1, 2]
min_d_bds = [0, 20, 50, 100]
tt, dd = np.meshgrid(min_t_bds, min_d_bds)