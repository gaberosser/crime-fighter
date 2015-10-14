__author__ = 'gabriel'
import numpy as np
import os
from scripts import OUT_DIR
import dill
import collections
import operator
from matplotlib import pyplot as plt
from analysis import chicago


INDIR = os.path.join(OUT_DIR, 'validate_chicago_stkde')

REGIONS = (
    'chicago_central',
    'chicago_southwest',
    'chicago_south',

    'chicago_far_southwest',
    'chicago_far_southeast',
    'chicago_west',

    'chicago_northwest',
    'chicago_north',
    'chicago_far_north',
)

CRIME_TYPES = (
    'burglary',
    'assault',
)

def load_one_result(region,
                    crime_type,
                    indir=INDIR,
                    include_model=False,
                    aggregate=True,
                    ):

    ind = os.path.join(indir, region, crime_type)
    this_file = os.path.join(ind, '{0}-{1}-validation.pickle').format(
        region, crime_type
    )
    res = {}
    try:
        with open(this_file, 'r') as f:
            vb = dill.load(f)
            if aggregate:
                res['cumulative_area'] = np.nanmean(vb['cumulative_area'], axis=0)
                res['cumulative_crime'] = np.nanmean(vb['cumulative_crime'], axis=0)
            else:
                res['cumulative_area'] = vb['cumulative_area']
                res['cumulative_crime'] = vb['cumulative_crime']
    except Exception as exc:
        # couldn't load data
        print exc
        pass
    # cast back to normal dictionary
    return dict(res)


def load_all_results(include_model=False,
                     aggregate=True):
    res = {}
    for r in REGIONS:
        res[r] = {}
        for ct in CRIME_TYPES:
            res[r][ct] = {'stkde': load_one_result(r, ct, indir=INDIR,
                                                   include_model=include_model,
                                                   aggregate=aggregate)
            }

    return res

