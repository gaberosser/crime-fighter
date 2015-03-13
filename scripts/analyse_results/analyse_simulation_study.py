__author__ = 'gabriel'
import mangle_data
import numpy as np
from matplotlib import pyplot as plt
from point_process import plotting
import dill


if __name__ == '__main__':
    sepp_objs = mangle_data.load_simulation_study()

    max_d = []
    max_t = []
    for k in sepp_objs.keys():
        max_t.append(k[0])
        max_d.append(k[1])
    max_t = np.unique(max_t)
    max_d = np.unique(max_d)

    lls = np.zeros((max_t.size, max_d.size))

    for (i, t), (j, d) in zip(enumerate(max_t), enumerate(max_d)):
        lls[i, j] = sepp_objs[(t, d)].log_likelihoods[-1]

