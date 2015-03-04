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

# fraction of crimes attributed to triggering (matrix)

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


# triggering plots

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
fixed_min_d = 0
styles = ('k-', 'k--', 'r-', 'r--', 'b-')
ts = np.linspace(0, 60, 500)
xs = np.linspace(-250, 250, 500)
for ct in prop_trigger.keys():
    fig = plt.figure('%s (time trigger)' % ct)
    # set bd_d=10
    for i, t in enumerate(min_t):
        this_sepp = sepp_objs[ct][(t, fixed_min_d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zt = trigger.marginal_pdf(ts, dim=0)
        plt.plot(ts, zt, styles[i])

    fig = plt.figure('%s (space trigger)' % ct)
    # set bd_t=0
    for i, t in enumerate(min_t):
        this_sepp = sepp_objs[ct][(t, fixed_min_d)]
        if not this_sepp:
            continue
        trigger = this_sepp.trigger_kde
        if not trigger:
            continue
        zx = trigger.marginal_pdf(xs, dim=1)
        plt.plot(xs, zx, styles[i])


# background plots

poly = mangle_data.load_boundary(location)
## fix min_t
fixed_min_t = 0
fixed_ct = 'burglary'

for i, d in enumerate(min_d):
    this_sepp = sepp_objs[fixed_ct][(fixed_min_t, d)]
    if not this_sepp:
        continue
    plotting.prediction_heatmap(this_sepp, t=0, kind='bgstatic', poly=poly)