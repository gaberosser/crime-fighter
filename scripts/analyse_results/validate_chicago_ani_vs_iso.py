__author__ = 'gabriel'
import numpy as np
import os
from .. import OUT_DIR
import dill
import collections
import operator
from matplotlib import pyplot as plt

INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl')
# INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl_keep_coincident')

REGIONS = (
    'chicago_central',
    'chicago_far_southwest',
    'chicago_northwest',
    'chicago_southwest',
    'chicago_far_southeast',
    'chicago_north',
    'chicago_south',
    'chicago_west',
    # 'chicago_far_north',
)

METHODS = (
    'ani',
    'iso',
    'ani_refl',
    'iso_refl',
)

CRIME_TYPES = (
    'burglary',
    'robbery',
    'assault',
    'motor_vehicle_theft',
)

def load_results_all_methods(region,
                             crime_type,
                             indir=INDIR,
                             include_model=False,
                             aggregate=True,
                             ):
    ind = os.path.join(indir, region, crime_type)
    res = collections.defaultdict(dict)
    for m in METHODS:
        this_file = os.path.join(ind, '{0}-{1}-{2}-validation.pickle').format(
            region, crime_type, m
        )
        try:
            with open(this_file, 'r') as f:
                vb = dill.load(f)
                r = vb['model'][0]
                if include_model:
                    res[m]['model'] = r
                if aggregate:
                    res[m]['cumulative_area'] = np.nanmean(vb['full_static']['cumulative_area'], axis=0)
                    res[m]['cumulative_crime'] = np.nanmean(vb['full_static']['cumulative_crime'], axis=0)
                else:
                    res[m]['cumulative_area'] = vb['full_static']['cumulative_area']
                    res[m]['cumulative_crime'] = vb['full_static']['cumulative_crime']
                res[m]['frac_trigger'] = r.num_trig[-1] / float(r.ndata)
        except Exception as exc:
            # couldn't load data
            print exc
            pass
    return res


def load_all_results(include_model=False):
    res = {}
    indirs = {
        'remove_coinc': INDIR,
        'keep_coinc': INDIR + '_keep_coincident',
    }
    for k, ind in indirs.iteritems():
        res[k] = {}
        for r in REGIONS:
            res[k][r] = {}
            for ct in CRIME_TYPES:
                res[k][r][ct] = load_results_all_methods(r, ct, indir=ind, include_model=include_model)

    return res


def pickle_all_models():
    res = load_all_results(include_model=True)
    indirs = {
        'remove_coinc': INDIR,
        'keep_coinc': INDIR + '_keep_coincident',
    }
    for k, ind in indirs.iteritems():
        for r in REGIONS:
            for ct in CRIME_TYPES:
                for m in METHODS:
                    try:
                        obj = res[k][r][ct][m]['model']
                        fn = os.path.join(ind, '{0}-{1}-{2}-model.pickle').format(
                            r, ct, m
                        )
                        obj.pickle(fn)
                    except Exception as exc:
                        print repr(exc)


if __name__ == '__main__':
    dist_between = 0.4
    s = 80
    crime_labels = [t.replace('_', ' ') for t in CRIME_TYPES]
    method_labels = [t.replace('_', ' ') for t in METHODS]

    INFILE = 'validate_chicago_ani_vs_iso.pickle'
    with open(INFILE, 'r') as f:
        res = dill.load(f)

    # category -> number
    crime_cat = dict([(k, i) for i, k in enumerate(CRIME_TYPES)])
    method_cat = dict([(k, i) for i, k in enumerate(METHODS)])

    for r in REGIONS:
        # data_coinc = np.zeros((len(CRIME_TYPES), len(METHODS)))
        # data_no_coinc = np.zeros((len(CRIME_TYPES), len(METHODS)))
        data_coinc = []
        data_no_coinc = []
        ct_v = []
        m_v = []
        for ct in CRIME_TYPES:
            for m in METHODS:
                # keeping coincident points
                ct_ix = crime_cat[ct]
                m_ix = method_cat[m]
                ct_v.append(ct_ix)
                m_v.append(m_ix)
                try:
                    a = res['keep_coinc'][r][ct][m]['frac_trigger']
                    if a == 0:
                        data_coinc.append(-1)
                    else:
                        data_coinc.append(a)
                except KeyError:
                    data_coinc.append(np.nan)
                try:
                    a = res['remove_coinc'][r][ct][m]['frac_trigger']
                    if a == 0:
                        data_no_coinc.append(-1)
                    else:
                        data_no_coinc.append(a)
                except KeyError:
                    data_no_coinc.append(np.nan)

        plt.figure()
        plt.scatter(np.array(ct_v) - dist_between/2., np.array(m_v), c=data_coinc, s=s, vmin=-1, vmax=1, cmap='RdYlBu')
        plt.scatter(np.array(ct_v) + dist_between/2., np.array(m_v), c=data_no_coinc, s=s, vmin=-1, vmax=1, cmap='RdYlBu')
        plt.title(r.replace('_', ' '))
        ax = plt.gca()
        xticks = reduce(operator.add, [[t - dist_between/2., t + dist_between/2.] for t in range(len(CRIME_TYPES))])
        xticklabels = reduce(operator.add, [['%s keep' % t, '%s remove' % t] for t in crime_labels])
        ax.set_xticks(xticks)
        ax.set_yticks(range(len(METHODS)))
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(method_labels)
        plt.colorbar()


