__author__ = 'gabriel'
from matplotlib import pyplot as plt
from scripts import OUT_DIR
import os
from bisect import bisect_left
import dill
import numpy as np


files = (
    'network_hit_rates.dill',
    'planar_grid_hit_rates.dill',
    'planar_net_hit_rates.dill',
)
subdir = os.path.join(OUT_DIR, 'birmingham')


def load_raw_data():
    x = []
    y = []
    xm = []
    ym = []
    for fn in files:
        with open(os.path.join(subdir, fn), 'r') as f:
            tmp = dill.load(f)
            x.append(tmp['x'])
            y.append(tmp['y'])
            xm.append(tmp['x'].mean(axis=0))
            ym.append(tmp['y'].mean(axis=0))
    return x, y, xm, ym


def wilcoxon_test(cmin=0.01, cmax=0.2, npt=500):
    from stats.pairwise import wilcoxon
    x, y, xm, ym = load_raw_data()
    n_net = xm[0].size
    n_grid = xm[2].size
    covs = np.linspace(cmin, cmax, npt)
    pvals = []
    outcomes = []
    for c in covs:
        idx_net = min(bisect_left(xm[0], c), n_net - 1)
        idx_grid = min(bisect_left(xm[2], c), n_grid - 1)
        ny = y[0][:, idx_net]
        gy = y[2][:, idx_grid]
        T, p, o = wilcoxon(ny, gy)
        pvals.append(p)
        outcomes.append(o)

    return np.array(pvals), np.array(outcomes)

if __name__ == '__main__':

    coverage_levels = (0.01, 0.05, 0.1)

    x = []
    y = []
    xm = []
    ym = []
    for fn in files:
        with open(os.path.join(subdir, fn), 'r') as f:
            tmp = dill.load(f)
            x.append(tmp['x'])
            y.append(tmp['y'])
            xm.append(tmp['x'].mean(axis=0))
            ym.append(tmp['y'].mean(axis=0))

    # fig: mean hit rates
    styles = ('k-', 'b-', 'r-')
    leg = ('Network', 'Planar-grid', 'Planar-network')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(3):
        ax.plot(xm[i], ym[i], styles[i])
    ax.legend(leg, loc='upper left', frameon=False)
    ax.set_xlim([0., 0.2])
    ax.set_ylim([0., 0.6])
    ax.set_xlabel('Coverage fraction')
    ax.set_ylabel('Fraction crime captured')

    # fig: improvement factor
    titles = ('1\% coverage',
              '5\% coverage',
              '10\% coverage')
    bins = np.linspace(-6, 6, 30)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    for i in range(3):
        idx_net = bisect_left(xm[0], coverage_levels[i])
        idx_grid = bisect_left(xm[2], coverage_levels[i])
        ny = y[0][:, idx_net]
        gy = y[2][:, idx_grid]
        b = ym[2][idx_grid]
        ax = axs[i]
        rp = (ny - gy) / b
        h = ax.hist(rp, bins)
        # add dashed line for mean
        c = ax.get_ylim()[1]
        d = rp.mean()
        ax.plot([d, d], [0, c], 'r--')

    axs[2].set_xlabel('Network:planar relative daily hit rate')
    axs[1].set_ylabel('Count')
    plt.tight_layout()


