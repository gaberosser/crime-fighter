__author__ = 'gabriel'
import numpy as np
import os
import dill
import csv
import scripts
import collections
from scripts import OUT_DIR


CRIME_TYPES = (
    'burglary',
    'theft_of_vehicle',
    'robbery',
    'violence'
)


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


def vres_at_coverage(vres, coverage=0.2, kind='full_static'):
    x = vres[kind]['cumulative_area']
    inds = [np.where(row>=coverage)[0][0] for row in x]
    hr = np.array([row[i] for row, i in zip(vres[kind]['cumulative_crime'], inds)])
    hr = hr[~np.isnan(hr)]
    pai = np.array([row[i] for row, i in zip(vres[kind]['pai'], inds)])
    pai = pai[~np.isnan(pai)]
    return hr, pai


def vres_means(vres, kind='full_static'):
    x = vres[kind]['cumulative_area'].mean(axis=0)
    hr = np.nanmean(vres[kind]['cumulative_crime'], axis=0)
    pai = np.nanmean(vres[kind]['pai'], axis=0)
    return x, hr, pai


def load_camden_min_bandwidths(coverage=0.2):

    hr = collections.defaultdict(dict)
    pai = collections.defaultdict(dict)
    kinds = ('full_static', 'bg_static', 'trigger')
    missing_data = []

    PARAMS_FILE = os.path.join(os.path.join(*scripts.__path__), 'parameters', 'vary_min_bandwidths.txt')
    with open(PARAMS_FILE, 'r') as f:
        c = csv.reader(f, delimiter=' ')
        for row in c:
            crime_type = row[0]
            if crime_type not in hr:
                hr[crime_type] = collections.defaultdict(dict)
                pai[crime_type] = collections.defaultdict(dict)
            min_t = float(row[1])
            min_d = float(row[2])
            in_file = os.path.join(OUT_DIR, 'camden', 'min_bandwidth', '%s_%.2f-%.2f-validation.pickle') % (
                crime_type,
                min_t,
                min_d
            )
            if not os.path.isfile(in_file):
                # no validation file
                log_file = os.path.join(OUT_DIR, 'camden', 'min_bandwidth', '%s-%.2f-%.2f.log') % (
                    crime_type,
                    min_t,
                    min_d
                )
                if os.path.isfile(log_file):
                    # failed run
                    for k in kinds:
                        hr[crime_type][k][(min_t, min_d)] = pai[crime_type][k][(min_t, min_d)] = None
                else:
                    # missing run
                    missing_data.append((crime_type, min_t, min_d))
            else:
                with open(in_file, 'r') as fi:
                    vres = dill.load(fi)
                    for k in kinds:
                        tmp = vres_at_coverage(vres, coverage=coverage, kind=k)
                        hr[crime_type][k][(min_t, min_d)], pai[crime_type][k][(min_t, min_d)] = tmp

    return hr, pai, missing_data


def load_min_bandwidths_mean_predictive_performance(location='camden', variant='min_bandwidth'):
    # alternative variant is 'min_bandwidth_trigger_only'
    a = collections.defaultdict(dict)
    hr = collections.defaultdict(dict)
    pai = collections.defaultdict(dict)
    kinds = ('full_static', 'bg_static', 'trigger')
    missing_data = []

    PARAMS_FILE = os.path.join(os.path.join(*scripts.__path__), 'parameters', 'vary_min_bandwidths.txt')
    with open(PARAMS_FILE, 'r') as f:
        c = csv.reader(f, delimiter=' ')
        for row in c:
            crime_type = row[0]
            if crime_type not in hr:
                a[crime_type] = collections.defaultdict(dict)
                hr[crime_type] = collections.defaultdict(dict)
                pai[crime_type] = collections.defaultdict(dict)
            min_t = float(row[1])
            min_d = float(row[2])
            in_file = os.path.join(OUT_DIR, location, variant, '%s_%.2f-%.2f-validation.pickle') % (
                crime_type,
                min_t,
                min_d
            )
            if not os.path.isfile(in_file):
                # no validation file
                log_file = os.path.join(OUT_DIR, location, variant, '%s-%.2f-%.2f.log') % (
                    crime_type,
                    min_t,
                    min_d
                )
                if os.path.isfile(log_file):
                    # failed run
                    for k in kinds:
                        hr[crime_type][k][(min_t, min_d)] = pai[crime_type][k][(min_t, min_d)] = None
                        a[crime_type][k][(min_t, min_d)] = None
                else:
                    # missing run
                    missing_data.append((crime_type, min_t, min_d))
            else:
                with open(in_file, 'r') as fi:
                    vres = dill.load(fi)
                    for k in kinds:
                        tmp = vres_means(vres, kind=k)
                        a[crime_type][k][(min_t, min_d)] = tmp[0]
                        hr[crime_type][k][(min_t, min_d)] = tmp[1]
                        pai[crime_type][k][(min_t, min_d)] = tmp[2]

    return a, hr, pai, missing_data


def load_prediction_results(res_path, params_file_path, format_fun, coverage=0.2):
    """
    Load results relating to predictive accuracy, etc.
    :param res_path: path to results folder, RELATIVE TO OUT_DIR
    :param params_file_path: path to relevant parameters file
    :param format_fun: the function used to convert from a set of parameters to a filename. The function takes the inputs
    (crime_type, *args), where args is the remaining parameters on a row.
    :return:
    """
    hr = collections.defaultdict(dict)
    pai = collections.defaultdict(dict)
    kinds = ('full_static', 'bg_static', 'trigger')
    missing_data = []
    with open(params_file_path, 'r') as f:
        c = csv.reader(f, delimiter=' ')
        for row in c:
            # assume first parameter is always crime type
            ct = row[0]
            if ct not in hr:
                # first time we've seen this crime type
                hr[ct] = collections.defaultdict(dict)
                pai[ct] = collections.defaultdict(dict)
            args = tuple(row[1:])
            in_file = os.path.join(OUT_DIR, res_path, format_fun(ct, *args))
            print in_file
            if not os.path.isfile(in_file):
                for k in kinds:
                    hr[ct][k][args] = pai[ct][k][args] = None
                    missing_data.append((ct,) + tuple(args))
            else:
                with open(in_file, 'r') as g:
                    vres = dill.load(g)
                    for k in kinds:
                        cov20 = vres_at_coverage(vres, kind=k, coverage=coverage) if vres is not None else (None, None)
                        hr[ct][k][args] = cov20[0]
                        pai[ct][k][args] = cov20[1]
    return hr, pai, missing_data


def load_trigger_background(location='camden', variant='min_bandwidth'):
    # alternative variant is 'min_bandwidth_trigger_only'
    PARAMS_FILE = os.path.join(os.path.join(*scripts.__path__), 'parameters', 'vary_min_bandwidths.txt')

    missing_data = []
    sepp_objs = {}
    proportion_trigger = {}

    with open(PARAMS_FILE, 'r') as f:
        c = csv.reader(f, delimiter=' ')
        for row in c:
            crime_type = row[0]
            if crime_type not in sepp_objs:
                sepp_objs[crime_type] = collections.defaultdict(dict)
                proportion_trigger[crime_type] = collections.defaultdict(dict)
            min_t = float(row[1])
            min_d = float(row[2])
            in_file = os.path.join(OUT_DIR, location, variant, '%s_%.2f-%.2f-vb_obj.pickle') % (
                crime_type,
                min_t,
                min_d
            )
            if not os.path.isfile(in_file):
                # no validation file
                log_file = os.path.join(OUT_DIR, location, variant, '%s-%.2f-%.2f.log') % (
                    crime_type,
                    min_t,
                    min_d
                )
                if os.path.isfile(log_file):
                    # failed run
                    sepp_objs[crime_type][(min_t, min_d)] = None
                    proportion_trigger[crime_type][(min_t, min_d)] = None
                else:
                    # missing run
                    missing_data.append((crime_type, min_t, min_d))
            else:
                with open(in_file, 'r') as fi:
                    obj = dill.load(fi)
                    sepp = obj.model
                    sepp_objs[crime_type][(min_t, min_d)] = sepp
                    proportion_trigger[crime_type][(min_t, min_d)] = sepp.p[sepp.linkage].sum() / float(sepp.ndata)

    return sepp_objs, proportion_trigger, missing_data


def load_boundary(location='camden'):
    PARAMS_FILE = os.path.join(os.path.join(*scripts.__path__), 'parameters', 'vary_min_bandwidths.txt')
    with open(PARAMS_FILE, 'r') as f:
        c = csv.reader(f, delimiter=' ')
        while True:
            row = c.next()
            crime_type = row[0]
            min_t = float(row[1])
            min_d = float(row[2])
            in_file = os.path.join(OUT_DIR, location, 'min_bandwidth', '%s_%.2f-%.2f-vb_obj.pickle') % (
                crime_type,
                min_t,
                min_d
            )
            if not os.path.isfile(in_file):
                continue
            with open(in_file, 'r') as fi:
                obj = dill.load(fi)
                if not obj:
                    continue
                return obj.roc.poly


def load_simulation_study():
    PARAMS_FILE = os.path.join(os.path.join(*scripts.__path__), 'parameters', 'simulation_vary_max_triggers.txt')
    with open(PARAMS_FILE, 'r') as f:
        c = csv.reader(f, delimiter=' ')
        sepp_objs = {}
        for row in c:
            max_t = float(row[0])
            max_d = float(row[1])
            in_file = os.path.join(OUT_DIR, 'simulation', 'simulation_%.2f-%.2f-sepp_obj.pickle') % (max_t, max_d)
            with open(in_file, 'r') as g:
                sepp_objs[(max_t, max_d)] = dill.load(g)

    return sepp_objs