__author__ = 'gabriel'
import dill
import numpy as np
import os
from matplotlib import pyplot as plt
from kde import optimisation
from scripts.optimal_bandwidth_birmingham_network import PARAM_EXTENT as NETWORK_PARAM, N_PT as NETWORK_NPT
from scripts.optimal_bandwidth_birmingham_planar import PARAM_EXTENT as PLANAR_PARAM, N_PT as PLANAR_NPT

RESULTS_DIR = '/home/gabriel/Dropbox/research/results/bandwidth_optimisation/birmingham/start_date_2013_07_01/'
NETWORK_DATA_DIR = os.path.join(RESULTS_DIR, 'network_linear_space_exponential_time')
NETWORK_FILE_LIST = (
    'birmingham_optimisation_start_day_180_10_iterations.dill',
    'birmingham_optimisation_start_day_190_10_iterations.dill',
    'birmingham_optimisation_start_day_200_10_iterations.dill',
    'birmingham_optimisation_start_day_210_10_iterations.dill',
    'birmingham_optimisation_start_day_220_10_iterations.dill',
    'birmingham_optimisation_start_day_230_10_iterations.dill',
)

PLANAR_DATA_DIR = os.path.join(RESULTS_DIR, 'planar_linear_space_exponential_time')
PLANAR_FILE_LIST = (
    'planar_linearexponentialkde_start_day_180_10_iterations.dill',
    'planar_linearexponentialkde_start_day_190_10_iterations.dill',
    'planar_linearexponentialkde_start_day_200_10_iterations.dill',
    'planar_linearexponentialkde_start_day_210_10_iterations.dill',
    'planar_linearexponentialkde_start_day_220_10_iterations.dill',
    'planar_linearexponentialkde_start_day_230_10_iterations.dill',
)

def load_all_chunks_network():
    n_per_file = 10
    z = []
    for fn in NETWORK_FILE_LIST:
        fullfile = os.path.join(NETWORK_DATA_DIR, fn)
        with open(fullfile, 'r') as f:
            x = dill.load(f)

        for j in range(n_per_file):
            z.append(np.array(x[j]))

    opt = optimisation.NetworkFixedBandwidth(None)
    opt.set_parameter_grid(NETWORK_NPT, *NETWORK_PARAM)
    bandwidths = np.array(list(opt.args_kwargs_generator()))
    t_grid = bandwidths[:, 0].reshape((NETWORK_NPT, NETWORK_NPT))
    d_grid = bandwidths[:, 1].reshape((NETWORK_NPT, NETWORK_NPT))
    return z, d_grid, t_grid


def load_all_chunks_planar():
    n_per_file = 10
    z = []
    t_grid = d_grid = None
    for fn in PLANAR_FILE_LIST:
        fullfile = os.path.join(PLANAR_DATA_DIR, fn)
        with open(fullfile, 'r') as f:
            x = dill.load(f)
        t_grid = x['tt']
        d_grid = x['dd']
        ll = x['ll']
        for j in range(n_per_file):
            z.append(np.array(ll[j]))
    return z, d_grid, t_grid


def combine_chunks(
        raw_values,
        shape=(100, 100),
        min_ll=np.log(1e-12)):
    """
     The original run was carried our in chunks of 10 to speed things up. Now we need to stitch them back together.
     The output is a flat list of len 10000, which represents the value at EVERY TARGET each day. We
     need to log and sum those, then reshape. In later runs, the output is a dict that includes the parameter grid.
    """
    ll = []

    for x in raw_values:
        this_ll = np.log(x)
        this_ll[this_ll < min_ll] = min_ll
        # reshape default order should be correct, as 'flat' was used originally
        this_ll = this_ll.sum(axis=1).reshape(shape)
        ll.append(this_ll)

    with open('start_date_2013_07_01_60_days.dill', 'w') as f:
        dill.dump(ll, f)

    return ll


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