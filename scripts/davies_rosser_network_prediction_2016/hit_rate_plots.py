__author__ = 'gabriel'
from matplotlib import pyplot as plt
from scripts import OUT_DIR
import os
import dill
import numpy as np
from validation import evaluation

TMAX = 90  # maximum prediction days

# custom results directory
OUT_DIR = '/home/gabriel/Dropbox/research/results/birmingham_jqc'

PLANAR_GRID = 'earlier_results/planar_grid_hit_rates.dill'
PLANAR_NET = 'earlier_results/planar_net_hit_rates.dill'
# NET = 'net_validation_start_day_240_90_iterations_end_hitrate.dill'
NET = 'net_validation_start_day_240_90_iterations_start_hitrate.dill'

# net files: for comparing the effect of different time windows
NET_FILES = (
    'net_validation_start_day_240_90_iterations_end_hitrate.dill',
    'net_validation_start_day_240_90_iterations_start_hitrate.dill',
    # 'net_validation_start_day_240_90_iterations_start_max24hours_hitrate.dill'
)

# compare files: for comparing net with planar
COMPARE_FILES = (
    NET,
    PLANAR_GRID,
    PLANAR_NET,
)

FONTSIZE = 18
# subdir = os.path.join(OUT_DIR, 'birmingham')


def load_time_window_data():
    x = []
    y = []
    xm = []
    ym = []
    for fn in NET_FILES:
        with open(os.path.join(OUT_DIR, fn), 'r') as f:
            tmp = dill.load(f)
            tmp['x'] = tmp['x'][:TMAX]
            tmp['y'] = tmp['y'][:TMAX]
            x.append(tmp['x'])
            y.append(tmp['y'])
            covs, means = evaluation.mean_hit_rate(tmp['x'], tmp['y'], n_covs=501)
            xm.append(covs)
            ym.append(means)
    return x, y, xm, ym


def load_compare_data():
    x = []
    y = []
    xm = []
    ym = []
    for fn in COMPARE_FILES:
        with open(os.path.join(OUT_DIR, fn), 'r') as f:
            tmp = dill.load(f)
            tmp['x'] = tmp['x'][:TMAX]
            tmp['y'] = tmp['y'][:TMAX]
            x.append(tmp['x'])
            y.append(tmp['y'])
            covs, means = evaluation.mean_hit_rate(tmp['x'], tmp['y'], n_covs=501)
            xm.append(covs)
            ym.append(means)
    return x, y, xm, ym


def wilcoxon_test(x0, y0, x1, y1, cmin=0.01, cmax=0.2, npt=500):
    from stats.pairwise import wilcoxon
    nday0, n0 = x0.shape
    nday1, n1 = x1.shape
    nday = min(nday0, nday1)
    covs = np.linspace(cmin, cmax, npt)
    pvals = []
    outcomes = []
    for c in covs:
        m0 = evaluation.interpolated_hit_rate(x0, y0, c)[:nday]
        m1 = evaluation.interpolated_hit_rate(x1, y1, c)[:nday]
        T, p, o = wilcoxon(m0, m1)
        pvals.append(p)
        outcomes.append(o)

    return covs, np.array(pvals), np.array(outcomes)


if __name__ == '__main__':

    coverage_levels = (0.02, 0.05, 0.1)
    sign_level = 0.05

    x, y, xm, ym = load_compare_data()
    covs, pvals, outcomes = wilcoxon_test(x[0], y[0], x[2], y[2])

    # fig: mean hit rates
    styles = ('k-', 'b-', 'r-')
    leg = ('Network', 'Planar')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in [0, 2]:  # net, planar-net
        ax.plot(xm[i], ym[i], styles[i])
    ax.legend(leg, loc='upper left', frameon=False, fontsize=FONTSIZE)
    ax.set_xlim([0., 0.2])
    ax.set_ylim([0., 0.6])
    ax.set_xlabel('Coverage fraction', size=FONTSIZE)
    ax.set_ylabel('Fraction crime captured', size=FONTSIZE)

    ax.fill_between(covs, 0, 1, (pvals < sign_level) & (outcomes == 1),
                    facecolor='k', edgecolor='none', alpha=0.2, interpolate=True)
    ax.fill_between(covs, 0, 1, (pvals < sign_level) & (outcomes == -1),
                    facecolor='r', edgecolor='none', alpha=0.2, interpolate=True)
    # [fig.savefig('mean_hit_rate_net_vs_planar.%s' % ext, dpi=300) for ext in ('png', 'pdf', 'eps')]

    # fig: improvement factor
    titles = ('2\%',
              '5\%',
              '10\%')
    bins = np.linspace(-6, 6, 30)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    for i in range(3):
        mnet = evaluation.interpolated_hit_rate(x[0], y[0], coverage_levels[i])[:TMAX]
        mplan = evaluation.interpolated_hit_rate(x[2], y[2], coverage_levels[i])[:TMAX]
        b = mplan.mean()
        rp = (mnet - mplan) / b

        ax = axs[i]
        h = ax.hist(rp, bins)
        # add dashed line for mean
        c = ax.get_ylim()[1]
        d = rp.mean()
        print "Coverage %f: %f relative hit rate" % (coverage_levels[i], d)

        ax.plot([d, d], [0, c], 'r--', label=titles[i])
        ax.text(0.1, 0.8, titles[i], fontsize=FONTSIZE,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)

    axs[2].set_xlabel('Network:planar relative daily hit rate', size=FONTSIZE)
    axs[1].set_ylabel('Count', size=16)
    plt.tight_layout()
    # [fig.savefig('relative_hit_rate_net_vs_planar.%s' % ext, dpi=300) for ext in ('png', 'pdf', 'eps')]

    x, y, xm, ym = load_time_window_data()

    # # fig: mean hit rates
    styles = ('k-', 'b-', 'r-')
    leg = ('End', 'Start')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(2):
        ax.plot(xm[i], ym[i], styles[i])
    ax.legend(leg, loc='upper left', frameon=False, fontsize=FONTSIZE)
    ax.set_xlim([0., 0.2])
    ax.set_ylim([0., 0.6])
    ax.set_xlabel('Coverage fraction', size=FONTSIZE)
    ax.set_ylabel('Fraction crime captured', size=FONTSIZE)

    # # fig: improvement factor
    titles = ('2\%',
              '5\%',
              '10\%')
    bins = np.linspace(-6, 6, 30)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    for i in range(3):
        mnet = evaluation.interpolated_hit_rate(x[0], y[0], coverage_levels[i])[:TMAX]
        mplan = evaluation.interpolated_hit_rate(x[1], y[1], coverage_levels[i])[:TMAX]
        b = mplan.mean()
        rp = (mnet - mplan) / b

        ax = axs[i]
        h = ax.hist(rp, bins)
        # add dashed line for mean
        c = ax.get_ylim()[1]
        d = rp.mean()
        print "Coverage %f: %f relative hit rate" % (coverage_levels[i], d)
        ax.plot([d, d], [0, c], 'r--')

    axs[2].set_xlabel('Relative daily hit rate', size=FONTSIZE)
    axs[1].set_ylabel('Count', size=FONTSIZE)
    plt.tight_layout()

    # x, y, xm, ym = load_compare_data()