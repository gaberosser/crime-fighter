__author__ = 'gabriel'
from scripts.analyse_results import mangle_data
import numpy as np
from matplotlib import pyplot as plt
from point_process import plotting
import os
import scripts

params_file_path = os.path.join(os.path.split(scripts.__file__)[0], 'parameters', 'vary_maximum_triggers.txt')
validation_format_fun = lambda ct, *args: '{0}_{1:d}-{2:d}-validation.pickle'.format(ct, *[int(t) for t in args])
sepp_format_fun = lambda ct, *args: '{0}_{1:d}-{2:d}-vb_obj.pickle'.format(ct, *[int(t) for t in args])

# location = 'camden'
location = 'chicago_south'

crime_types = ['burglary', 'robbery', 'theft_of_vehicle', 'violence']
max_t = [7, 14, 21, 30, 60, 90, 120, 150]
max_d = [50, 100, 200, 300, 400, 500, 1000]
tt, dd = np.meshgrid(max_t, max_d)

init_bg_frac = {
    10: os.path.join(location, 'max_triggers_grid250_bgfrac10'),
    50: os.path.join(location, 'max_triggers_grid250_bgfrac50'),
    90: os.path.join(location, 'max_triggers_grid250_bgfrac90'),
}

pai20 = {}
missing = {}
sepp = {}

# load all data

for bg_prop, res_path in init_bg_frac.items():
    _, pai20[bg_prop], _ = mangle_data.load_prediction_results(res_path, params_file_path, validation_format_fun)
    sepp[bg_prop], missing[bg_prop] = mangle_data.load_sepp_objects(res_path, params_file_path, sepp_format_fun)

# plots for single crime type

ct = 'burglary'

for bg_prop in init_bg_frac:

    ##  Log likelihood array

    fig, axes = plt.subplots(nrows=len(max_d), ncols=len(max_t), sharex=True, sharey=True, num='%s ll array %d' % (ct, bg_prop))
    for i in range(len(max_t)):
        for j in range(len(max_d)):
            key = (str(max_t[i]), str(max_d[j]))
            if key in sepp[bg_prop][ct] and sepp[bg_prop][ct][key] is not None:
                this_ll = sepp[bg_prop][ct][key].log_likelihoods
                axes[len(max_d) - j - 1, i].plot(range(1, len(this_ll) + 1), this_ll)

    for i in range(len(max_t)):
        axes[-1, i].set_xlabel(max_t[i])

    for j in range(len(max_d)):
        axes[len(max_d) - j - 1, 0].set_ylabel(max_d[j])


    # prop trigger/BG array

    fig, axes = plt.subplots(nrows=len(max_d), ncols=len(max_t), sharex=True, sharey=True, num='%s prop array %d' % (ct, bg_prop))
    for i in range(len(max_t)):
        for j in range(len(max_d)):
            key = (str(max_t[i]), str(max_d[j]))
            if key in sepp[bg_prop][ct] and sepp[bg_prop][ct][key] is not None:
                this_prop_trig = np.array(sepp[bg_prop][ct][key].num_trig) / float(sepp[bg_prop][ct][key].ndata)
                axes[len(max_d) - j - 1, i].plot(range(1, len(this_prop_trig) + 1), 1 - this_prop_trig,
                                                 range(1, len(this_prop_trig) + 1), this_prop_trig)

    for i in range(len(max_t)):
        axes[-1, i].set_xlabel(max_t[i])

    for j in range(len(max_d)):
        axes[len(max_d) - j - 1, 0].set_ylabel(max_d[j])

## number of linkages
ct = 'burglary'
bg_prop = 10
n_linkage = np.zeros_like(tt)

for i in range(len(max_t)):
    for j in range(len(max_d)):
        key = (str(max_t[i]), str(max_d[j]))
        if key in sepp[bg_prop][ct] and sepp[bg_prop][ct][key] is not None:
            n_linkage[j, i] = sepp[bg_prop][ct][key].linkage[0].size
            ndata = sepp[bg_prop][ct][key].ndata
            n_linkage_max = ndata * (ndata + 1) / 2.0

## cpu time
log_format_fun = lambda ct, *args: '{0}_{1:d}-{2:d}.log'.format(ct, *[int(t) for t in args])
train_time = {}
pred_time = {}

for bg_prop, res_path in init_bg_frac.items():
    train_time[bg_prop], pred_time[bg_prop] = mangle_data.load_computation_time(res_path, params_file_path, log_format_fun)

# ct = 'burglary'
# ct = 'robbery'
ct = 'violence'
bg_prop = 50

t_styles = [
    'd',
    's',
    '^',
    'o',
    'x',
    's',
    '+',
    '*',
]

ax_train = plt.figure().add_subplot(111)
ax_pred = plt.figure().add_subplot(111)

x = range(1, len(max_d) + 1)
labels = []
for i in range(len(max_t))[::-1]:
    this_train_time = []
    this_pred_time = []
    for j in range(len(max_d)):
        key = (str(max_t[i]), str(max_d[j]))
        if key in train_time[ct]:
            this_train_time.append(train_time[ct][key].mean())
        else:
            this_train_time.append(np.nan)
        if key in pred_time[ct]:
            this_pred_time.append(pred_time[ct][key].mean())
        else:
            this_pred_time.append(np.nan)
    ax_train.plot(x, this_train_time,
                  marker=t_styles[i],
                  color='k',
                  markersize=15,
                  markerfacecolor='none',
                  markeredgewidth=1.5,
                  linestyle='none')
    ax_pred.plot(x, this_pred_time,
                  marker=t_styles[i],
                  color='k',
                  markersize=15,
                  markerfacecolor='none',
                  markeredgewidth=1.5,
                  linestyle='none')
    labels.append(r"$\Delta t_{\mathrm{max}} = %d$" % max_t[i])

ax_train.legend(labels, loc=2, numpoints=1)
ax_train.set_xlim([0.5, len(max_t) - 0.5])
ax_train.set_xticks(x)
ax_train.set_xticklabels(max_t, fontsize=16)
ax_train.set_xlabel(r'$\Delta d_{\mathrm{max}}$ (m)', fontsize=16)
ax_train.set_ylabel('Computation time per iteration (s)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

ax_pred.legend(labels, loc=2, numpoints=1)
ax_pred.set_xlim([0.5, len(max_t) - 0.5])
ax_pred.set_xticks(x)
ax_pred.set_xticklabels(max_t, fontsize=16)
ax_pred.set_xlabel(r'$\Delta d_{\mathrm{max}}$ (m)', fontsize=16)
ax_pred.set_ylabel('Computation time per prediction (s)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)



# log likelihoods

stationary_idx = -10
bg_prop = 50

lls = {}
for ct in crime_types:
    if ct not in lls:
        lls[ct] = {}
    for k, v in sepp[bg_prop][ct].iteritems():
        if v is not None:
            lls[ct][tuple(int(t) for t in k)] = np.mean(v.log_likelihoods[stationary_idx:])

for ct in crime_types:
    this_ll = lls[ct]
    zz = np.zeros_like(tt, dtype=float)
    for i, (t, d) in enumerate(zip(tt.flat, dd.flat)):
        if (t, d) in this_ll:
            zz.flat[i] = this_ll[(t, d)]
        else:
            zz.flat[i] = np.nan
    fig = plt.figure('%s-ll' % ct)
    ax = fig.add_subplot(111)
    h = ax.imshow(zz, interpolation='none', origin='lower')
    plt.colorbar(h)
    ax.set_xticks(range(len(max_t)))
    ax.set_yticks(range(len(max_t)))
    ax.set_xticklabels(max_t, fontsize=16)
    ax.set_yticklabels(max_d, fontsize=16)


d_styles = {
    50:  'k-',
    100: 'r-',
    200: 'b-',
    300: 'g-',
    400: 'c-',
    500: 'm-',
    1000:'y-'
}

t_styles = {
    7:  'k-',
    14: 'r-',
    21: 'b-',
    30: 'g-',
    60: 'c-',
    90: 'm-',
    120:'y-',
    150:'k--',
}

for ct in crime_types:
    fig = plt.figure('%s ll by t' % ct)
    ax = fig.add_subplot(111)
    for k, v in sepp[ct].iteritems():
        if k[1] == "2000":
            continue
        if v is not None:
            ax.plot(v.log_likelihoods, t_styles[int(k[0])])

for ct in crime_types:
    fig = plt.figure('%s ll by d' % ct)
    ax = fig.add_subplot(111)
    for k, v in sepp[ct].iteritems():
        if k[1] == "2000":
            continue
        if v is not None:
            ax.plot(v.log_likelihoods, d_styles[int(k[1])])

for ct in crime_types:
    fig, axes = plt.subplots(nrows=len(max_d), ncols=len(max_t), sharex=True, sharey=True, num='%s ll array' % ct)
    for i in range(len(max_t)):
        for j in range(len(max_d)):
            key = (str(max_t[i]), str(max_d[j]))
            if key in sepp[ct] and sepp[ct][key] is not None:
                this_ll = sepp[ct][key].log_likelihoods
                axes[len(max_d) - j - 1, i].plot(range(1, len(this_ll) + 1), this_ll)

    for i in range(len(max_t)):
        axes[-1, i].set_xlabel(max_t[i])

    for j in range(len(max_d)):
        axes[len(max_d) - j - 1, 0].set_ylabel(max_d[j])


# proportion trigger

ct = 'burglary'
vmax = 0.5
this_bg_prop = 50
prop_trig = {}

zz = np.zeros_like(tt, dtype=float)
for i in range(len(max_t)):
    for j in range(len(max_d)):
        v = sepp[this_bg_prop][ct].get((str(max_t[i]), str(max_d[j])), None)
        if v is not None:
            zz[j, i] = (v.ndata - v.p.diagonal().sum()) / float(v.ndata)
        else:
            zz[j, i] = np.nan

fig = plt.figure('%s-prop_trigger' % ct)
ax = fig.add_subplot(111)
h = ax.imshow(zz, interpolation='none', origin='lower')
plt.colorbar(h)
ax.set_xticks(range(len(max_t)))
ax.set_yticks(range(len(max_t)))
ax.set_xticklabels(max_t, fontsize=16)
ax.set_yticklabels(max_d, fontsize=16)

for ct in crime_types:
    if ct not in prop_trig:
        prop_trig[ct] = {}
    for k, v in sepp[ct].iteritems():
        if v is not None:
            prop_trig[ct][tuple(int(t) for t in k)] = (v.ndata - v.p.diagonal().sum()) / float(v.ndata)

for ct in crime_types:
    this_pt = prop_trig[ct]
    zz = np.zeros_like(tt, dtype=float)
    for i, (t, d) in enumerate(zip(tt.flat, dd.flat)):
        if (t, d) in this_pt:
            zz.flat[i] = this_pt[(t, d)]
        else:
            zz.flat[i] = np.nan
    fig = plt.figure('%s-prop_trigger' % ct)
    ax = fig.add_subplot(111)
    h = ax.imshow(zz, interpolation='none', origin='lower')
    plt.colorbar(h)
    ax.set_xticks(range(len(max_t)))
    ax.set_yticks(range(len(max_t)))
    ax.set_xticklabels(max_t, fontsize=16)
    ax.set_yticklabels(max_d, fontsize=16)

for ct in crime_types:
    fig, axes = plt.subplots(nrows=len(max_d), ncols=len(max_t), sharex=True, sharey=True, num='%s prop. trigger array 2' % ct)
    for i in range(len(max_t)):
        for j in range(len(max_d)):
            key = (str(max_t[i]), str(max_d[j]))
            if key in sepp[ct] and sepp[ct][key] is not None:
                this_prop_trig = np.array(sepp[ct][key].num_trig) / float(sepp[ct][key].ndata)
                axes[len(max_d) - j - 1, i].plot(range(1, len(this_prop_trig) + 1), 1 - this_prop_trig,
                                                 range(1, len(this_prop_trig) + 1), this_prop_trig)

    for i in range(len(max_t)):
        axes[-1, i].set_xlabel(max_t[i])

    for j in range(len(max_d)):
        axes[len(max_d) - j - 1, 0].set_ylabel(max_d[j])


# triggering function

fixed_t = 90
bg_prop = 50
styles = ('k-', 'k--', 'r-', 'r--', 'b-', 'b--', 'g-', 'g--')
d = np.linspace(-100, 100, 2000)
t = np.linspace(0, fixed_t, 2000)
for ct in crime_types:
    fig = plt.figure('%s-triggering' % ct)
    axt = fig.add_subplot(1, 2, 1)
    axd = fig.add_subplot(1, 2, 2)
    for k, v in sepp[bg_prop][ct].iteritems():
        if k[1] == "2000":
            continue
        if v is None:
            continue
        kde_obj = v.trigger_kde
        if kde_obj is None:
            continue
        yt = kde_obj.marginal_pdf(t, dim=0, normed=False)
        yx = kde_obj.marginal_pdf(d, dim=1, normed=False)
        axt.plot(t, yt, d_styles[int(k[1])])
        axd.plot(d, np.log(yx), d_styles[int(k[1])])
    axd.set_yscale('log')
    axd.set_ylim([0.01, 10])

# predictive accuracy
bg_prop = 50
mpai = {}
spai = {}
for ct in crime_types:
    if ct not in mpai:
        mpai[ct] = {}
        spai[ct] = {}
    for k, v in pai20[bg_prop][ct].iteritems():
        if v is not None:
            mpai[ct][tuple(int(t) for t in k)] = np.nanmean(v)
            spai[ct][tuple(int(t) for t in k)] = np.nanstd(v, ddof=1)
