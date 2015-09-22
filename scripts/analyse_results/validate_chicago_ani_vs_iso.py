__author__ = 'gabriel'
import numpy as np
import os
from .. import OUT_DIR
import dill
import collections

INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl')
METHODS = (
    'ani',
    'iso',
    'ani_refl',
    'iso_refl',
)


def load_results_all_methods(region, crime_type):
    indir = os.path.join(INDIR, region, crime_type)
    res = collections.defaultdict(dict)
    for m in METHODS:
        this_file = os.path.join(indir, '{0}-{1}-{2}-validation.pickle').format(
            region, crime_type, m
        )
        try:
            with open(this_file, 'r') as f:
                vb = dill.load(f)
                res[m]['model'] = vb['model'][0]
                res[m]['cumulative_area'] = vb['full_static']['cumulative_area'].mean(axis=0)
                res[m]['cumulative_crime'] = vb['full_static']['cumulative_crime'].mean(axis=0)
        except Exception as exc:
            # couldn't load data
            print exc
            pass
    return res