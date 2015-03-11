__author__ = 'gabriel'
import mangle_data
import numpy as np
from matplotlib import pyplot as plt
from point_process import plotting

# location = 'camden'
location = 'chicago_south'

sepp_objs, prop_trigger, missing_data = mangle_data.load_trigger_background(location=location)
# sepp_objs_to, prop_trigger_to, missing_data_to = mangle_data.load_trigger_background(
#     location=location, variant='min_bandwidth_trigger_only')

min_t = [0, .25, .5, 1, 2]
min_d = [0, 10, 20, 50, 100]
tt, dd = np.meshgrid(min_t, min_d)

## fraction of crimes attributed to triggering (matrix)

trigger_frac = {}
# trigger_frac_to = {}
for ct in prop_trigger.keys():
    zz = np.zeros_like(tt)
    # zz_to = np.zeros_like(tt)
    for i in range(zz.size):
        t = tt.flat[i]; d = dd.flat[i]
        zz.flat[i] = prop_trigger[ct][(t, d)]
        # zz_to.flat[i] = prop_trigger_to[ct][(t, d)]
    trigger_frac[ct] = zz
    # trigger_frac_to[ct] = zz_to


## triggering plots

# fix min_t = 0, vary min_d
fixed_min_t = 0.

styles = ('y-', 'k--', 'r-', 'r--', 'b-')
ts = np.linspace(0, 60, 500)
xs = np.linspace(-250, 250, 500)
for ct in prop_trigger.keys():
    fig = plt.figure('%s (time trigger)' % ct)
    # set bd_t=0
    for i, d in enumerate(min_d):
        this_sepp = sepp_objs[ct][(fixed_min_t, d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zt = trigger.marginal_pdf(ts, dim=0)
        plt.plot(ts, zt, styles[i])

    fig = plt.figure('%s (space trigger)' % ct)
    # set bd_t=0
    for i, d in enumerate(min_d):
        this_sepp = sepp_objs[ct][(fixed_min_t, d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zx = trigger.marginal_pdf(xs, dim=1)
        plt.plot(xs, zx, styles[i])

## fix min_d = 10, vary min_t
styles = (
    {'color': 'k',
     'ls': '-',
     'alpha': 0.4,},
    {'color': 'k',
     'ls': '--',
     'alpha': 0.8},
    {'color': 'r',
     'ls': '-',
     'alpha': 0.4,},
    {'color': 'r',
     'ls': '--',
     'alpha': 0.8,},
    {'color': 'b',
     'ls': '-',
     'alpha': 0.4,},
)

fixed_min_d = 10

ts = np.linspace(0, 60, 500)
xs = np.linspace(-250, 250, 500)
for ct in prop_trigger.keys():
    fig = plt.figure('%s (time trigger)' % ct)
    for i, t in enumerate(min_t):
        this_sepp = sepp_objs[ct][(t, fixed_min_d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zt = trigger.marginal_pdf(ts, dim=0)
        plt.plot(ts, zt, **styles[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Trigger density')
    plt.legend([r'$t_\mathrm{min}=%.2f$' % t for t in min_t])

    fig = plt.figure('%s (x trigger)' % ct)
    for i, t in enumerate(min_t):
        this_sepp = sepp_objs[ct][(t, fixed_min_d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zx = trigger.marginal_pdf(xs, dim=1)
        plt.plot(xs, zx, **styles[i])
    plt.xlabel('Distance (m)')
    plt.ylabel('Trigger density')

    fig = plt.figure('%s (y trigger)' % ct)
    for i, t in enumerate(min_t):
        this_sepp = sepp_objs[ct][(t, fixed_min_d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zy = trigger.marginal_pdf(xs, dim=2)
        plt.plot(xs, zy, **styles[i])
    plt.xlabel('Distance (m)')
    plt.ylabel('Trigger density')

## background plots

poly = mangle_data.load_boundary(location)
## fix min_t
fixed_min_t = 0
fixed_ct = 'burglary'

for i, d in enumerate(min_d):
    this_sepp = sepp_objs[fixed_ct][(fixed_min_t, d)]
    if not this_sepp:
        continue
    plotting.prediction_heatmap(this_sepp, t=0, kind='bgstatic', poly=poly)


## likelihoods

# plots

styles_by_d = {
    0: 'k-',
    10: 'r-',
    20: 'g-',
    50: 'k--',
    100: 'r--'
}

styles_by_t = {
    0: 'k-',
    0.25: 'k--',
    0.5: 'r-',
    1: 'r--',
    2: 'b'
}

ll_max = {}
ll_min = {}
ll_range = {}

for ct in sepp_objs.keys():
    fig = plt.figure(ct)
    ax = fig.add_subplot(111)

    ll_max[ct] = np.max([np.nanmax(v.log_likelihoods) for v in sepp_objs[ct].values() if v is not None])
    ll_min[ct] = np.min([np.nanmin(v.log_likelihoods) for v in sepp_objs[ct].values() if v is not None])
    ll_range[ct] = ll_max[ct] - ll_min[ct]
    leg = []

    for k, v in sorted(sepp_objs[ct].items(), key=lambda x: x[0][1]):
        if v:
            line = ax.plot((np.array(v.log_likelihoods) - ll_min[ct]) / ll_range[ct], styles_by_d[k[1]])
            # line = ax.plot((np.array(v.log_likelihoods) - ll_min[ct]) / ll_range[ct], styles_by_t[k[0]])
            if k[0] == 0:
                line[0].set_label(r'$h_{d,\mathrm{min}}=%d$' % k[1])
            ax.set_xlabel('Iteration', fontsize=14)
            ax.set_ylabel('Log likelihood (AU)', fontsize=14)

        # ax.legend()



stationary_idx = 25  # point after which the LL is stationary - need to generate plots as in previous code to find this

lls = {}

for ct in sepp_objs.keys():
    lls[ct] = {}

    for k, v in sepp_objs[ct].iteritems():
        if v:
            data = np.array(v.log_likelihoods[stationary_idx:])
            mstd = (data.mean(), data.std(ddof=1))
            lls[ct][k] = mstd


############################

# aggregate over values of min_t
styles_by_d = {
    0: 'k-',
    10: 'r-',
    20: 'b-',
    50: 'g-',
    100: 'c-'
}

fig = plt.figure("ll")
ax = fig.add_subplot(111)
i = 1
# di = 0.2
for ct in sepp_objs.keys():
    max_ll = -1e10
    min_ll = 0
    m = {}
    s = {}
    for d in min_d:
        obs = []
        for t in min_t:
            obs.extend(sepp_objs[ct][(t, d)].log_likelihoods if sepp_objs[ct][(t, d)] else [])

        obs = np.array(obs)
        if not len(obs):
            continue
        m[d] = np.nanmean(obs)
        s[d] = np.nanstd(obs, ddof=1)
        max_ll = max(max_ll, np.nanmax(obs))
        min_ll = min(min_ll, np.nanmin(obs))

    for d in min_d:
        if d not in m:
            continue
        m[d] = (m[d] - min_ll) / (max_ll - min_ll)
        s[d] /= (max_ll - min_ll)
        # hbar = ax.bar(i - di/4, 2 * s[d], bottom=m[d] - s[d], color=styles_by_d[d][0], width=di/2)
        hbar = ax.bar(i, 2 * s[d], bottom=m[d] - s[d], color=styles_by_d[d][0], width=1)
        if ct == 'violence':
            hbar.set_label(r'$h_{d,\mathrm{min}}=%d$' % d)
        i += 1
    # i += di
    i += 2

# ax.set_xlim([1 - di, 1 + 6 * di])
ax.set_xlim([0, 28])
ax.set_ylim([-0.2, 1.2])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([r'$0\%$', r'$50\%$', r'$100\%$'])
# ax.set_xticks([1 + n * di for n in range(4)])
ax.set_xticks([3.5, 9.5, 15.5, 21.5])
ax.set_xticklabels([t.replace('_', ' ') for t in sepp_objs.keys()], rotation=45)
ax.set_ylabel('Normalised log likelihood', fontsize=16)
ax.set_xlabel('Crime type', fontsize=16)
ax.legend(loc=4)
ax.set_position([0.12, 0.24, 0.96, 0.96])


############################


# prediction accuracy - look for significant differences using the Wilcoxon Signed-Rank test
# a more refined method is BEST by Kruschke

from scipy import stats, sparse

a, hr, pai, missing = mangle_data.load_min_bandwidths_mean_predictive_performance(location=location)
# hr, pai, missing = mangle_data.load_camden_min_bandwidths()
sig_level = 0.05
min_t = [0, .25, .5, 1, 2]
min_d = [0, 10, 20, 50, 100]
tt, dd = np.meshgrid(min_t, min_d)

diff_pai_20_by_t = {}
diff_pai_20_by_t_map = {}
for ct in hr.keys():
    diff_pai_20_by_t[ct] = {}
    diff_pai_20_by_t_map[ct] = {}
    for i, t in enumerate(min_t):
        zz = sparse.csr_matrix((len(min_d), len(min_d)))
        # every possible min_d combination
        for j, d0 in enumerate(min_d):
            for k in range(j+1, len(min_d)):
                d1 = min_d[k]
                x0 = pai[ct]['full_static'][(t, d0)]
                x1 = pai[ct]['full_static'][(t, d1)]
                if x0 is None or x1 is None:
                    continue
                zz[j, k] = stats.wilcoxon(x1, x0)[1]
                # std_err = (x1 - x0).std(ddof=1) / np.sqrt(x0.size)
                # if std_err == 0:
                #     continue
                # tstat = (x1 - x0).mean() / std_err
                # zz[j, k] = stats.t.sf(np.abs(tstat), x0.size - 1)

        # check for significance using Bonferroni's correction
        bonferroni_sig_level = (1 - (1 - sig_level) ** zz.nnz) / float(zz.nnz)
        diff_pai_20_by_t_map[ct][t] = np.where((zz.toarray() < bonferroni_sig_level) & (zz.toarray() != 0))
        diff_pai_20_by_t[ct][t] = zz

diff_pai_20_by_d = {}
diff_pai_20_by_d_map = {}
for ct in hr.keys():
    diff_pai_20_by_d[ct] = {}
    diff_pai_20_by_d_map[ct] = {}
    for i, d in enumerate(min_d):
        zz = sparse.csr_matrix((len(min_t), len(min_t)))
        # every possible min_d combination
        for j, t0 in enumerate(min_t):
            for k in range(j+1, len(min_t)):
                t1 = min_t[k]
                x0 = pai[ct]['full_static'][(t0, d)]
                x1 = pai[ct]['full_static'][(t1, d)]
                if x0 is None or x1 is None:
                    continue
                try:
                    zz[j, k] = stats.wilcoxon(x1, x0)[1]
                except ValueError:
                    # this occurs when the two arrays are identical
                    print "Identical arrays %s, %s" % (str((t0, d)), str((t1, d)))
                    pass

        diff_pai_20_by_d[ct][d] = zz
        # check for significance using Bonferroni's correction
        if zz.nnz:
            bonferroni_sig_level = (1 - (1 - sig_level) ** zz.nnz) / float(zz.nnz)
            diff_pai_20_by_d_map[ct][d] = np.where((zz.toarray() < bonferroni_sig_level) & (zz.toarray() != 0))


# search for significant differences (one-sided)

