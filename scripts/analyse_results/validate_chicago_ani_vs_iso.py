__author__ = 'gabriel'
import numpy as np
import os
from scripts import OUT_DIR
import dill
import collections
import operator
from matplotlib import pyplot as plt
from analysis import chicago
from tools import get_ellipse_coords


# INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl')
INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl_keep_coincident')

REGIONS = (
    'chicago_central',
    'chicago_far_southwest',
    'chicago_northwest',
    'chicago_southwest',
    'chicago_far_southeast',
    'chicago_north',
    'chicago_south',
    'chicago_west',
    'chicago_far_north',
)

METHODS = (
    # 'ani',
    # 'iso',
    'ani_refl',
    'iso_refl',
    'ani_norm',
)

CRIME_TYPES = (
    'burglary',
    # 'robbery',
    'assault',
    # 'motor_vehicle_theft',
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
    # cast back to normal dictionary
    return dict(res)


def load_all_results(include_model=False,
                     aggregate=True):
    res = {}
    for r in REGIONS:
        res[r] = {}
        for ct in CRIME_TYPES:
            res[r][ct] = load_results_all_methods(r, ct, indir=INDIR,
                                                  include_model=include_model,
                                                  aggregate=aggregate)

    return res


def pickle_all_models():
    res = load_all_results(include_model=True)
    for r in REGIONS:
        for ct in CRIME_TYPES:
            for m in METHODS:
                try:
                    obj = res[r][ct][m]['model']
                    fn = os.path.join(INDIR, '{0}-{1}-{2}-model.pickle').format(
                        r, ct, m
                    )
                    obj.pickle(fn)
                except Exception as exc:
                    print repr(exc)


def wilcoxon_analysis(this_res):
    from stats.pairwise import wilcoxon
    out = {}
    for m1 in METHODS:
        for m2 in METHODS:
            if (m2, m1) in out:
                continue
            try:
                area = np.nanmean(this_res[m1]['cumulative_area'], axis=0)
                n = area.size
                t = []
                for i in range(n):
                    x1 = this_res[m1]['cumulative_crime'][:, i]
                    x2 = this_res[m2]['cumulative_crime'][:, i]
                    a, b = wilcoxon(x1, x2)
                    t.append({'area': area[i],
                              'Z': a,
                              'pval': b})
                out[(m1, m2)] = t
            except Exception:
                pass
    return out


def plot_mean_hit_rate(this_res, cutoff=0.2, ax=None, legend=True):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    colour_mapping = {
        'ani_norm': 'k',
        'ani_refl': 'r',
        'iso_refl': 'b',
    }
    for m in METHODS:
        try:
            c = colour_mapping[m]
            x = np.nanmean(this_res[m]['cumulative_area'], axis=0)
            y = np.nanmean(this_res[m]['cumulative_crime'], axis=0)
            ax.plot(x[x<=cutoff], y[x<=cutoff], colour_mapping[m])
        except Exception as exc:
            print repr(exc)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Hit rate')
    if legend:
        ax.legend([t.replace('_', ' ') for t in METHODS], loc='upper left')


if __name__ == '__main__':
    dist_between = 0.4
    s = 80
    crime_labels = [t.replace('_', ' ') for t in CRIME_TYPES]
    method_labels = [t.replace('_', ' ') for t in METHODS]

    # INFILE = 'validate_chicago_ani_vs_iso.pickle'
    # with open(INFILE, 'r') as f:
    #     res = dill.load(f)

    # category -> number
    crime_cat = dict([(k, i) for i, k in enumerate(CRIME_TYPES)])
    method_cat = dict([(k, i) for i, k in enumerate(METHODS)])

    # for r in REGIONS:
    #     data_coinc = []
    #     data_no_coinc = []
    #     ct_v = []
    #     m_v = []
    #     for ct in CRIME_TYPES:
    #         for m in METHODS:
    #             # keeping coincident points
    #             ct_ix = crime_cat[ct]
    #             m_ix = method_cat[m]
    #             ct_v.append(ct_ix)
    #             m_v.append(m_ix)
    #             try:
    #                 a = res['keep_coinc'][r][ct][m]['frac_trigger']
    #                 if a == 0:
    #                     data_coinc.append(-1)
    #                 else:
    #                     data_coinc.append(a)
    #             except KeyError:
    #                 data_coinc.append(np.nan)
    #             try:
    #                 a = res['remove_coinc'][r][ct][m]['frac_trigger']
    #                 if a == 0:
    #                     data_no_coinc.append(-1)
    #                 else:
    #                     data_no_coinc.append(a)
    #             except KeyError:
    #                 data_no_coinc.append(np.nan)
    #
    #     plt.figure()
    #     plt.scatter(np.array(ct_v) - dist_between/2., np.array(m_v), c=data_coinc, s=s, vmin=-1, vmax=1, cmap='RdYlBu')
    #     plt.scatter(np.array(ct_v) + dist_between/2., np.array(m_v), c=data_no_coinc, s=s, vmin=-1, vmax=1, cmap='RdYlBu')
    #     plt.title(r.replace('_', ' '))
    #     ax = plt.gca()
    #     xticks = reduce(operator.add, [[t - dist_between/2., t + dist_between/2.] for t in range(len(CRIME_TYPES))])
    #     xticklabels = reduce(operator.add, [['%s keep' % t, '%s remove' % t] for t in crime_labels])
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(range(len(METHODS)))
    #     ax.set_xticklabels(xticklabels)
    #     ax.set_yticklabels(method_labels)
    #     plt.colorbar()

    bar_width = 0.3

    for ct in CRIME_TYPES:

        frac_trigger_iso = []
        frac_trigger_ani = []
        frac_trigger_nor = []
        x_pos_iso = []
        x_pos_ani = []
        x_pos_nor = []
        x_pos_all = []
        x_label = []
        i = 1.
        for r in REGIONS:
            try:
                frac_trigger_ani.append(res[r][ct]['ani_refl']['frac_trigger'])
            except Exception as exc:
                print repr(exc)
                frac_trigger_ani.append(np.nan)
            try:
                frac_trigger_iso.append(res[r][ct]['iso_refl']['frac_trigger'])
            except Exception as exc:
                print repr(exc)
                frac_trigger_iso.append(np.nan)
            try:
                frac_trigger_nor.append(res[r][ct]['ani_norm']['frac_trigger'])
            except Exception as exc:
                print repr(exc)
                frac_trigger_nor.append(np.nan)
            x_pos_ani.append(i)
            x_pos_iso.append(i + bar_width)
            x_pos_nor.append(i + 2 * bar_width)

            x_pos_all.append(i + 1.5 * bar_width)

            x_label.append(r.replace('_', ' ').capitalize())
            i += 1

        plt.figure()
        plt.bar(x_pos_ani, frac_trigger_ani, bar_width, color='r', alpha=0.4)
        plt.bar(x_pos_iso, frac_trigger_iso, bar_width, color='b', alpha=0.4)
        plt.bar(x_pos_nor, frac_trigger_nor, bar_width, color='g', alpha=0.4)
        ax = plt.gca()
        ax.set_xticks(np.array(x_pos_all))
        ax.set_xticklabels(x_label)
        plt.title(ct.replace('_', ' ').capitalize())
        plt.legend((
            'Ani, refl',
            'Iso, refl',
            'Ani, norm',
        ))

    domains = chicago.get_chicago_side_polys(as_shapely=True)
    domain_mapping = {
        'chicago_south': 'South',
        'chicago_southwest': 'Southwest',
        'chicago_west': 'West',
        'chicago_northwest': 'Northwest',
        'chicago_north': 'North',
        'chicago_central': 'Central',
        'chicago_far_north': 'Far North',
        'chicago_far_southwest': 'Far Southwest',
        'chicago_far_southeast': 'Far Southeast',
    }

    colour_mapping = {
        'ani_norm': 'k',
        'ani_refl': 'r',
        'iso_refl': 'b',
    }

    t = 0.95

    for ct in CRIME_TYPES:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
        i = 0
        for r in REGIONS:
            dom = domains[domain_mapping[r]]
            for m in METHODS:
                try:
                    k = res[r][ct][m]['model'].trigger_kde
                    if k.ndim == 3:
                        a = k.marginal_icdf(t, dim=1)
                        b = k.marginal_icdf(t, dim=2)
                        coords = get_ellipse_coords(a=a, b=b)
                    else:
                        a = k.marginal_icdf(t, dim=1)
                        coords = get_ellipse_coords(a, a)
                    axs.flat[i].plot(coords[:, 0], coords[:, 1], colour_mapping[m])

                except Exception as exc:
                    print repr(exc)
            axs.flat[i].title.set_text(domain_mapping[r])
            i += 1


    for ct in CRIME_TYPES:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
        i = 0
        for r in REGIONS:
            plot_mean_hit_rate(res[r][ct], ax=axs.flat[i], legend=(i == 0))
            axs.flat[i].set_title(domain_mapping[r])
            i += 1


    from plotting.spatiotemporal import pairwise_distance_histogram
    for ct in CRIME_TYPES:
        for r in REGIONS:
            try:
                data = res[r][ct]['ani_refl']['model'].data
                pairwise_distance_histogram(data, max_t=120, max_d=500, fmax=0.99)
                plt.title("%s - %s" % (domain_mapping[r], ct.capitalize()))
            except Exception as exc:
                print repr(exc)
