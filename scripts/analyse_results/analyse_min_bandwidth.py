__author__ = 'gabriel'
import mangle_data
import numpy as np
from matplotlib import pyplot as plt
from point_process import plotting

# location = 'camden'
location = 'chicago_south'

sepp_objs, prop_trigger, missing_data = mangle_data.load_trigger_background(location=location)
sepp_objs_to, prop_trigger_to, missing_data_to = mangle_data.load_trigger_background(
    location=location, variant='min_bandwidth_trigger_only')

min_t = [0, .25, .5, 1, 2]
min_d = [0, 10, 20, 50, 100]
tt, dd = np.meshgrid(min_t, min_d)

## fraction of crimes attributed to triggering (matrix)

trigger_frac = {}
trigger_frac_to = {}
for ct in prop_trigger.keys():
    zz = np.zeros_like(tt)
    zz_to = np.zeros_like(tt)
    for i in range(zz.size):
        t = tt.flat[i]; d = dd.flat[i]
        zz.flat[i] = prop_trigger[ct][(t, d)]
        zz_to.flat[i] = prop_trigger_to[ct][(t, d)]
    trigger_frac[ct] = zz
    trigger_frac_to[ct] = zz_to


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
for ct in sepp_objs.keys():
    fig = plt.figure(ct)
    ax = fig.add_subplot(111)
    for k, v in sepp_objs[ct].iteritems():
        if v:
            ax.plot(v.log_likelihoods)


stationary_idx = 25  # point after which the LL is stationary - need to generate plots as in previous code to find this

lls = {}
for ct in sepp_objs.keys():
    lls[ct] = {}
    for k, v in sepp_objs[ct].iteritems():
        if v:
            data = np.array(v.log_likelihoods[stationary_idx:])
            lls[ct][k] = (data.mean(), data.std(ddof=1))
    sorted(lls[ct].items(), key=lambda x: x[1][0])

# prediction accuracy - look for significant differences using the paired sample t-test
# a more refined method is BEST by Kruschke

from scipy import stats

hr, pai, missing = mangle_data.load_camden_min_bandwidths()
min_t = [0, .25, .5, 1, 2]
min_d = [0, 10, 20, 50, 100]
tt, dd = np.meshgrid(min_t, min_d)

diff_pai_20 = {}
for ct in hr.keys():
    diff_pai_20[ct] = {}
    for i, t in enumerate(min_t):
        zz = np.zeros((len(min_d), len(min_d)))
        for j, d0 in enumerate(min_d):
            for k in range(j+1, len(min_d)):
                d1 = min_d[k]
                x0 = pai[ct]['full_static'][(t, d0)]
                x1 = pai[ct]['full_static'][(t, d1)]
                if x0 is None or x1 is None:
                    continue
                std_err = (x1 - x0).std(ddof=1) / np.sqrt(x0.size)
                if std_err == 0:
                    continue
                tstat = (x1 - x0).mean() / std_err
                zz[j, k] = stats.t.sf(np.abs(tstat), x0.size - 1)
            # zz[i, j] = t
        diff_pai_20[ct][t] = zz