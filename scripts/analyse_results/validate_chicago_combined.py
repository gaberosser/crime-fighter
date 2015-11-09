__author__ = 'gabriel'
import numpy as np
import os
from validate_chicago_ani_vs_iso import load_all_results as load_sepp
from validate_chicago_stkde import load_all_results as load_stkde
from database.chicago.consts import REGIONS, FILE_FRIENDLY_REGIONS, ABBREVIATED_REGIONS
import dill
import collections
import operator
from matplotlib import pyplot as plt
from analysis import chicago
from tools import get_ellipse_coords


CRIME_TYPES = (
    'burglary',
    'assault',
)


def recursive_update(the_dict, dict_to_add):
    """
    Update the_dict with contents of dict_to_add RECURSIVELY, traversing sub-dicts and updating them too
    :param the_dict:
    :param dict_to_add:
    :return:
    """
    items_to_add = {}
    for k, v in dict_to_add.iteritems():
        if k in the_dict and isinstance(the_dict[k], dict):
            assert isinstance(v, dict), "Trying to update a sub-dict with non-dict object"
            recursive_update(the_dict[k], v)
        else:
            items_to_add[k] = v
    the_dict.update(items_to_add)


def load_all_results(aggregate=False):
    # sepp_res = load_sepp(aggregate=aggregate, include_model=False)
    with open('/home/gabriel/Dropbox/research/output/chicago_ani_vs_iso/hit_rates_full.pickle', 'r') as f:
        sepp_res = dill.load(f)
    stkde_res = load_stkde(aggregate=aggregate)
    res_all = sepp_res
    # recursive_update(res_all, sepp_res)
    recursive_update(res_all, stkde_res)

    return res_all


def plot_wilcox_test_hit_rate(this_res,
                              ax=None,
                              sign_level=0.05,
                              max_cover=0.2,
                              min_difference=0.01):
    """
    Plot the mean hit rate vs coverage along with an overlay that indicates significant differences between methods.
    Wilcox test is run on ani_refl and iso_refl.
    Mean STKDE result is included for reference.
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
    y_kde_mean = np.nanmean(this_res['stkde']['cumulative_crime'][:, idx], axis=0)
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
    ax.plot(x, y_kde_mean, 'b--')
    ax.fill_between(x, 0, 1, (pvals < sign_level) & (outcomes == 1) & ((y_iso_mean - y_ani_mean) >= min_difference),
                    facecolor='k', edgecolor='none', alpha=0.3, interpolate=True)
    ax.fill_between(x, 0, 1, (pvals < sign_level) & (outcomes == -1) & ((y_ani_mean - y_iso_mean) >= min_difference),
                    facecolor='r', edgecolor='none', alpha=0.3, interpolate=True)
    # TODO: make this more rigorous. Add darker shading for double min_difference
    ax.fill_between(x, 0, 1, (pvals < sign_level) & (outcomes == 1) & ((y_iso_mean - y_ani_mean) >= 2 * min_difference),
                    facecolor='k', edgecolor='none', alpha=0.3, interpolate=True)
    ax.fill_between(x, 0, 1, (pvals < sign_level) & (outcomes == -1) & ((y_ani_mean - y_iso_mean) >= 2 * min_difference),
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



if __name__ == '__main__':
    pass