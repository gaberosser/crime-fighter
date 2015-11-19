__author__ = 'gabriel'
import dill
import os
from scripts import OUT_DIR
from matplotlib import pyplot as plt
import numpy as np

all_results = {
    'camden': (
        'burglary',
        'violence',
        'shoplifting',
    ),
    'chicago_south': (
        'burglary',
        'assault',
        'motor_vehicle_theft'
    )
}

RESULTS_DIR = os.path.join(OUT_DIR, 'network_vs_planar')

def hit_rate_plots(cumulative_coverage,
                   cumulative_crime,
                   xmax=0.2,
                   methods=None):
    """
    :param cumulative_coverage: Dict of dicts [crime_type][method] containing MEAN data
    :param cumulative_crime:
    :param xmax:
    :param methods: If supplied, only these methods are included
    :return:
    """
    styles = ['k-', 'b-', 'r-']
    n_ct = len(cumulative_coverage)
    fig, axs = plt.subplots(1, n_ct, sharex=True, sharey=True)

    for i, ct in enumerate(cumulative_crime.keys()):
        ax = axs[i]
        for j, m in enumerate(cumulative_crime[ct].keys()):
            if methods is not None and m not in methods:
                continue
            x = cumulative_coverage[ct][m]
            y = cumulative_crime[ct][m]
            ax.plot(x, y, styles[j])
        ax.set_title(ct.replace('_', ' ').capitalize())
        ax.set_xlim([0, xmax])
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Mean hit rate')

        if i == (n_ct - 1):
            ax.legend([t.replace('_', ' ') for t in cumulative_crime[ct].keys()], loc='se')

    plt.tight_layout()



if __name__ == '__main__':
    cumulative_coverage_mean = {}
    cumulative_crime_all = {}
    cumulative_crime_mean = {}
    cumulative_crime_20pct = {}

    for loc, crime_types in all_results.items():
        cumulative_crime_all[loc] = {}
        cumulative_crime_mean[loc] = {}
        cumulative_crime_20pct[loc] = {}
        cumulative_coverage_mean[loc] = {}
        for ct in crime_types:
            infile = os.path.join(RESULTS_DIR, '%s_%s.pickle' % (loc, ct))
            with open(infile, 'r') as f:
                res = dill.load(f)
            cumulative_crime_all[loc][ct] = {}
            cumulative_crime_mean[loc][ct] = {}
            cumulative_crime_20pct[loc][ct] = {}
            cumulative_coverage_mean[loc][ct] = {}
            for method, val in res.iteritems():
                x = cumulative_coverage_mean[loc][ct][method] = val['cumulative_area'].mean(axis=0)
                cumulative_crime_all[loc][ct][method] = val['cumulative_crime']
                y = cumulative_crime_mean[loc][ct][method] = val['cumulative_crime'].mean(axis=0)
                idx20pct = np.where(x <= 0.2)[0][-1]
                cumulative_crime_20pct[loc][ct][method] = y[idx20pct]
