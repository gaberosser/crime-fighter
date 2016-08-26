__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.special import erfinv
from scripts import OUT_DIR
import os
import dill
from validation import evaluation

TMAX = 90  # maximum prediction days

# custom results directory
OUT_DIR = '/home/gabriel/Dropbox/research/results/birmingham_jqc'

PLANAR_GRID = 'earlier_results/planar_grid_hit_rates.dill'
PLANAR_NET = 'earlier_results/planar_net_hit_rates.dill'
NET = 'net_validation_start_day_240_90_iterations_end_hitrate.dill'

# net files: for comparing the effect of different time windows
NET_FILES = (
    'net_validation_start_day_240_90_iterations_end_hitrate.dill',
    'net_validation_start_day_240_90_iterations_start_hitrate.dill',
    'net_validation_start_day_240_90_iterations_start_max24hours_hitrate.dill'
)

# compare files: for comparing net with planar
COMPARE_FILES = (
    NET,
    PLANAR_GRID,
    PLANAR_NET,
)

FONTSIZE = 18


def load_compare_data():
    x = []
    y = []
    xm = []
    ym = []
    for fn in COMPARE_FILES:
        with open(os.path.join(OUT_DIR, fn), 'r') as f:
            tmp = dill.load(f)
            tmp['x'] = tmp['x'][:TMAX]
            tmp['y'] = tmp['y'][:TMAX]
            x.append(tmp['x'])
            y.append(tmp['y'])
            covs, means = evaluation.mean_hit_rate(tmp['x'], tmp['y'], n_covs=501)
            xm.append(covs)
            ym.append(means)
    return x, y, xm, ym


def load_time_window_data():
    x = []
    y = []
    xm = []
    ym = []
    for fn in NET_FILES:
        with open(os.path.join(OUT_DIR, fn), 'r') as f:
            tmp = dill.load(f)
            tmp['x'] = tmp['x'][:TMAX]
            tmp['y'] = tmp['y'][:TMAX]
            x.append(tmp['x'])
            y.append(tmp['y'])
            covs, means = evaluation.mean_hit_rate(tmp['x'], tmp['y'], n_covs=501)
            xm.append(covs)
            ym.append(means)
    return x, y, xm, ym


if __name__ == '__main__':
    x, y, xm, ym = load_compare_data()
    m0 = evaluation.interpolated_hit_rate(x[0], y[0], 0.05)
    m1 = evaluation.interpolated_hit_rate(x[2], y[2], 0.05)
    d = m0 - m1

    acorr = acf(d, nlags=21)
    # Confidence interval: z value / sqrt(sample size)
    ci = np.sqrt(2) * erfinv(0.95) / np.sqrt(d.size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acorr, 'k-')
    ax.plot([0, len(acorr)], [ci, ci], 'k--')
    ax.plot([0, len(acorr)], [-ci, -ci], 'k--')
    plt.xlabel('Lag (days)', size=FONTSIZE)
    plt.ylabel('Autocorrelation', size=FONTSIZE)
    plt.tight_layout()