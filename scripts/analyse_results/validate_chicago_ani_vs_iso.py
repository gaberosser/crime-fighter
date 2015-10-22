__author__ = 'gabriel'
import numpy as np
import os
from scripts import OUT_DIR
import dill
import collections
import operator
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
from analysis import chicago
from tools import get_ellipse_coords
from database.chicago.consts import REGIONS, FILE_FRIENDLY_REGIONS, ABBREVIATED_REGIONS


# INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl')
INDIR = os.path.join(OUT_DIR, 'validate_chicago_ani_vs_iso_refl_keep_coincident')

REGIONS_OLD = (
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

METHODS = (
    'ani_refl',
    'iso_refl',
    # 'ani_norm',
)

CRIME_TYPES = (
    'burglary',
    'assault',
)

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
    for r in REGIONS_OLD:
        res[r] = {}
        for ct in CRIME_TYPES:
            res[r][ct] = load_results_all_methods(r, ct, indir=INDIR,
                                                  include_model=include_model,
                                                  aggregate=aggregate)

    return res


def pickle_all_models():
    res = load_all_results(include_model=True)
    for r in REGIONS_OLD:
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


def bar_triggering_fractions(res, save=True):
    method_map = ('ani_refl', 'iso_refl')
    bar_pad = 0.05
    for ct in CRIME_TYPES:
        y = []
        for r in REGIONS:
            for m in method_map:
                try:
                    y.append(res[FILE_FRIENDLY_REGIONS[r]][ct][m]['frac_trigger'])
                except KeyError:
                    y.append(None)
        y = np.array(y, dtype=float)
        x = np.arange(len(y)) / 2.
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(x[::2] + bar_pad, y[::2], 0.5 * (1 - 2 * bar_pad), fc='k')
        ax.bar(x[1::2], y[1::2], 0.5 * (1 - 2 * bar_pad), fc='b')
        ax.set_xticks(x[1::2])
        ax.set_xticklabels([ABBREVIATED_REGIONS[r] for r in REGIONS], rotation=45)
        ax.set_ylabel('Fraction triggering')
        ax.legend(('Anisotropic', 'Isotropic'), loc='upper right')
        ax.set_ylim([0, 1.])
        plt.tight_layout()

        if save:
            filename = "chicago_%s_proportion_triggering" % ct
            fig.savefig(filename + '.png', dpi=200)
            fig.savefig(filename + '.pdf')
        


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


def plot_wilcox_test_hit_rate(this_res, ax=None,
                              sign_level=0.05,
                              max_cover=0.2,
                              min_difference=0.01):
    """
    Plot the mean hit rate vs coverage along with an overlay that indicates significant differences between methods.
    :param this_res:
    :param ax:
    :param sign_level: The two-tailed significance level
    :param max_cover:
    :param min_difference: The minimum magnitude of difference required before an overlay is plotted
    :return:
    """
    from stats import pairwise
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    x = np.nanmean(this_res['ani_refl']['cumulative_area'], axis=0)
    idx = x <= max_cover
    x = x[idx]
    y_ani = this_res['ani_refl']['cumulative_crime'][:, idx]
    y_iso = this_res['iso_refl']['cumulative_crime'][:, idx]
    y_ani_mean = np.nanmean(y_ani, axis=0)
    y_iso_mean = np.nanmean(y_iso, axis=0)
    pvals = []
    outcomes = []
    for i in range(y_ani.shape[1]):
        W, p, out = pairwise.wilcoxon(y_iso[:, i], y_ani[:, i])
        pvals.append(p)
        outcomes.append(out)
    pvals = np.array(pvals)
    outcomes = np.array(outcomes)
    ax.plot(x, y_ani_mean, 'r')
    ax.plot(x, y_iso_mean, 'k')
    ax.fill_between(x, 0, 1, (pvals < sign_level) & (outcomes == 1) & ((y_iso_mean - y_ani_mean) >= min_difference),
                    facecolor='k', edgecolor='none', alpha=0.3, interpolate=True)
    ax.fill_between(x, 0, 1, (pvals < sign_level) & (outcomes == -1) & ((y_ani_mean - y_iso_mean) >= min_difference),
                    facecolor='r', edgecolor='none', alpha=0.3, interpolate=True)


def plot_hit_rate_array(res, max_cover=0.2, min_difference=0.01, save=True):
    domains = chicago.get_chicago_side_polys(as_shapely=True)
    chicago_poly = chicago.compute_chicago_region(as_shapely=True).simplify(1000)

    for ct in CRIME_TYPES:

        fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharex=False, sharey=False)

        for i, r in enumerate(REGIONS):
            ax_j = np.mod(i, 3)
            ax_i = i / 3
            ax = axs[ax_i, ax_j]

            this_res = res[FILE_FRIENDLY_REGIONS[r]][ct]
            try:
                plot_wilcox_test_hit_rate(this_res, ax=ax, max_cover=max_cover, min_difference=min_difference)
            except KeyError:
                continue

            if ax_i != 2:
                ax.set_xticks([])
            if ax_j != 0:
                ax.set_yticks([])

        for i in range(len(REGIONS)):
            axs.flat[i].set_xlim([0., max_cover])
            axs.flat[i].set_ylim([0., 1.])

        big_ax = fig.add_subplot(111)
        big_ax.spines['top'].set_color('none')
        big_ax.spines['bottom'].set_color('none')
        big_ax.spines['left'].set_color('none')
        big_ax.spines['right'].set_color('none')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        big_ax.set_xlabel('Coverage')
        big_ax.set_ylabel('Hit rate')
        big_ax.patch.set_visible(False)

        plt.tight_layout(pad=1.5, h_pad=0.25, w_pad=0.5)
        big_ax.set_position([0.05, 0.05, 0.95, 0.9])

        inset_pad = 0.01  # proportion of parent axis width or height
        inset_width_ratio = 0.35
        inset_height_ratio = 0.55

        for i, r in enumerate(REGIONS):
            ax_j = np.mod(i, 3)
            ax_i = i / 3
            ax = axs[ax_i, ax_j]
            ax_bbox = ax.get_position()
            inset_bbox = [
                ax_bbox.x0 + inset_pad * ax_bbox.width,
                ax_bbox.y1 - (inset_pad + inset_height_ratio) * ax_bbox.height,
                inset_width_ratio * ax_bbox.width,
                inset_height_ratio * ax_bbox.height,
            ]
            inset_ax = fig.add_axes(inset_bbox)
            chicago.plot_domain(chicago_poly, sub_domain=domains[r], ax=inset_ax)

        if save:
            filename = 'chicago_ani_iso_kde_hit_rate_%s' % ct
            fig.savefig(filename + '.png', dpi=200)
            fig.savefig(filename + '.pdf')


def plot_spatial_ellipse_array(res, icdf=0.95, max_d=800., save=True):
    """
    Plot the ellipsoids containing icdf of the spatial triggering density
    :param res: Results dict, including model
    :param icdf: The proportion of spatial density in each dimension contained within the ellipse boundary
    :param max_d: The axis maximum value. This is increased automatically if it is too small to contain any of the
    ellipses.
    :return:
    """
    icdf_two_tailed = 0.5 + icdf / 2.
    domains = chicago.get_chicago_side_polys(as_shapely=True)
    colour_mapping = {
        # 'ani_norm': 'k',
        'ani_refl': 'r',
        'iso_refl': 'k',
    }
    loc = plticker.MultipleLocator(base=400.0) # this locator puts ticks at regular intervals
    abbreviated_regions = {
        'South': 'S',
        'Southwest': 'SW',
        'West': 'W',
        'Northwest': 'NW',
        'North': 'N',
        'Central': 'C',
        'Far North': 'FN',
        'Far Southwest': 'FSW',
        'Far Southeast': 'FSE',
    }

    for ct in CRIME_TYPES:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))
        for i, r in enumerate(REGIONS):
            ax_j = np.mod(i, 3)
            ax_i = i / 3
            ax = axs[ax_i, ax_j]
            dom = domains[r]
            for m in ('ani_refl', 'iso_refl'):
                try:
                    k = res[FILE_FRIENDLY_REGIONS[r]][ct][m]['model'].trigger_kde
                    if k.ndim == 3:
                        a = k.marginal_icdf(icdf_two_tailed, dim=1)
                        b = k.marginal_icdf(icdf_two_tailed, dim=2)
                        coords = get_ellipse_coords(a=a, b=b)
                        max_d = max(max(a, b), max_d)
                    else:
                        a = k.marginal_icdf(icdf, dim=1)
                        coords = get_ellipse_coords(a, a)
                        max_d = max(a, max_d)
                    ax.plot(coords[:, 0], coords[:, 1], colour_mapping[m])

                except Exception as exc:
                    print repr(exc)
            ax.text(0.04, 0.86, abbreviated_regions[r], fontsize=14, transform=ax.transAxes)

        for ax in axs.flat:
            ax.set_xlim([-max_d, max_d])
            ax.set_ylim([-max_d, max_d])
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)
            ax.set_aspect('equal', adjustable='box-forced')

        plt.tight_layout(h_pad=0.2, w_pad=0.2)

        big_ax = fig.add_subplot(111)
        big_ax.spines['top'].set_color('none')
        big_ax.spines['bottom'].set_color('none')
        big_ax.spines['left'].set_color('none')
        big_ax.spines['right'].set_color('none')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        big_ax.set_xlabel(r'$\Delta x$')
        big_ax.set_ylabel(r'$\Delta y$')
        big_ax.patch.set_visible(False)

        big_ax.set_position([0.05, 0.05, 0.95, 0.9])

        if save:
            filename = "chicago_triggering_spatial_ellipse_%s" % ct
            fig.savefig(filename + '.png', dpi=200)
            fig.savefig(filename + '.pdf')


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
        for r in REGIONS_OLD:
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
        for r in REGIONS_OLD:
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
        for r in REGIONS_OLD:
            plot_mean_hit_rate(res[r][ct], ax=axs.flat[i], legend=(i == 0))
            axs.flat[i].set_title(domain_mapping[r])
            i += 1


    from plotting.spatiotemporal import pairwise_distance_histogram
    for ct in CRIME_TYPES:
        for r in REGIONS_OLD:
            try:
                data = res[r][ct]['ani_refl']['model'].data
                pairwise_distance_histogram(data, max_t=120, max_d=500, fmax=0.99)
                plt.title("%s - %s" % (domain_mapping[r], ct.capitalize()))
            except Exception as exc:
                print repr(exc)
