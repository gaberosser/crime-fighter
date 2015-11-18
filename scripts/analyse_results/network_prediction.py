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
