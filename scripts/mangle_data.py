__author__ = 'gabriel'
import numpy as np
import os
import dill
from . import OUT_DIR


def validation_full(vres, coverage=0.2):
    '''
    :param vres: output from validation 'run' method
    '''
    out = {}
    for k, res in vres.iteritems():
        if not res:
            # skip unconverged
            out[k] = None
            continue
        x = res['full']['cumulative_area'].mean(axis=0)
        idx = np.where(x >= coverage)[0][0]
        out[k] = {
            'hit_rate': res['full']['cumulative_crime'][:, idx],
            'pai': res['full']['pai'][:, idx],
        }

    return out

def validation_summary(vres, coverage=0.2):
    '''
    :param vres: output from validation 'run' method
    :return:
    '''
    out = {}

    vfull = validation_full(vres, coverage=coverage)
    for k, res in vfull.iteritems():
        if res is None:
            out[k] = None
            continue
        out[k] = {
            'hit_rate': (np.nanmean(res['hit_rate']), np.nanstd(res['hit_rate'])),
            'pai': (np.nanmean(res['pai']), np.nanstd(res['pai'])),
        }
    return out


def load_camden_validation_evaluation(coverage=0.2):

    res = {}
    names = ('burglary', 'violence', 'robbery', 'theft_of_vehicle')

    for name in names:
        with open(os.path.join(OUT_DIR, 'camden', name, 'validation.pickle'), 'r') as f:
            vres = dill.load(f)
            res[name] = validation_full(vres, coverage=coverage)

    return res


def load_chicago_south_ageing(coverage=0.2):

    out = {}
    names = ('burglary', 'violence', 'robbery', 'theft_of_vehicle')
    base_dir = os.path.join(OUT_DIR, 'chicago', 'model_ageing')

    for name in names:
        with open(os.path.join(base_dir, name, 'validation.pickle'), 'r') as f:
            vres = dill.load(f)
        tmp = validation_full(vres)
        out[name] = dict([(k, val) for k, val in tmp.iteritems() if val is not None])
    return out

