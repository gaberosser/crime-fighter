__author__ = 'gabriel'
import dill
import numpy as np
import os
from matplotlib import pyplot as plt


def load_all_chunks():
    start_days = range(180, 240, 10)
    n_per_file = 10
    data_dir = '/home/gabriel/Dropbox/research/results/bandwidth_optimisation/birmingham/start_date_2013_07_01/'

    z = []

    for i in start_days:
        filename = 'birmingham_optimisation_start_day_%d_10_iterations.dill' % i
        fullfile = os.path.join(data_dir, filename)
        with open(fullfile, 'r') as f:
            x = dill.load(f)
        for j in range(n_per_file):
            z.append(np.array(x[j]))

    return z


def log_likelihood_hist(raw_values):
    """
    :param raw_values: From load_all_chunks
    :return:
    """
    non_z = []
    for x in raw_values:
        non_z.extend(x[x != 0.])
    plt.hist(np.log(non_z), 50)
    plt.xlabel('Single crime log likelihood')
    plt.ylabel('Count')


def combine_chunks(min_ll=np.log(1e-12)):
    """
     The original run was carried our in chunks of 10 to speed things up. Now we need to stitch them back together.
     In addition, the output is a flat list of len 10000, which represents the value at EVERY TARGET each day. We
     need to log and sum those, then reshape.
    """

    # minimum ll: to avoid -inf when the value is zero, we set this minimum before summing lls
    start_days = range(180, 240, 10)
    n_per_file = 10
    data_dir = '/home/gabriel/Dropbox/research/results/bandwidth_optimisation/birmingham/start_date_2013_07_01/'

    # get param grid
    from scripts.optimal_bandwidth_birmingham import PARAM_EXTENT, N_PT, optimisation
    opt = optimisation.NetworkFixedBandwidth(None)
    opt.set_parameter_grid(N_PT, *PARAM_EXTENT)
    bandwidths = np.array(list(opt.args_kwargs_generator()))
    tt = bandwidths[:, 0].reshape((N_PT, N_PT))
    dd = bandwidths[:, 1].reshape((N_PT, N_PT))

    raw_values = load_all_chunks()

    ll = []

    for x in raw_values:
        this_ll = np.log(x)
        this_ll[this_ll < min_ll] = min_ll
        # reshape default order should be correct, as 'flat' was used originally
        this_ll = this_ll.sum(axis=1).reshape((100, 100))
        ll.append(this_ll)

    with open('start_date_2013_07_01_60_days.dill', 'w') as f:
        dill.dump(ll, f)

    return ll, dd, tt


def sum_surface_plot(ll, tt, dd, fmin=0.5, cmap='Reds'):

    """
    NB if cmap has non-white low end, need some extra work to get the right BG colour
    """

    z = np.zeros_like(ll[0])
    for l in ll:
        z += l

    vmin = sorted(z.flat)[int(fmin * z.size)]
    vmax = z.max()
    colour_bins = np.linspace(vmin, vmax, 100)
    cmap = plt.get_cmap(cmap)

    plt.figure()
    plt.contourf(tt, dd, z, colour_bins, cmap=cmap)
    plt.colorbar()
    plt.xlabel("Time bandwidth (days)")
    plt.ylabel("Network distance bandwidth (days)")

    i, j = np.unravel_index(np.argmax(z), z.shape)
    topt = tt[i, j]
    dopt = dd[i, j]
    plt.plot([topt, topt], [0, dopt], 'k--')
    plt.plot([0, topt], [dopt, dopt], 'k--')
