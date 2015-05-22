__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
from data.models import DataArray
from random import shuffle
from point_process import simulate
from scipy import stats

## assume SEPP object is stored in variable r

x_range = y_range = [-r.max_delta_d, r.max_delta_d]
npt = 200
xy = DataArray.from_meshgrid(*np.meshgrid(
    np.linspace(x_range[0], x_range[1], npt),
    np.linspace(y_range[0], y_range[1], npt),
))

rows = cols = 5
trigger_idx = np.where([t is not None for t in r.trigger_kde])[0]
shuffle(trigger_idx)

fig = plt.figure()
scatter_ax = fig.add_subplot(111)
scatter_ax.scatter(r.data.getdim(1), r.data.getdim(2), marker='o', c='k', s=r.data.getdim(0) / max(r.data.toarray(0)) * 30, edgecolor='none', alpha=0.3)
scatter_ax.set_aspect('equal')

fig, axes = plt.subplots(rows, cols, sharex='all', sharey='all')

for i in range(rows * cols):
    scatter_ax.plot(r.data[trigger_idx[i], 1], r.data[trigger_idx[i], 2], 'rx', markersize=20, lw=4)
    scatter_ax.text(r.data[trigger_idx[i], 1], r.data[trigger_idx[i], 2], '%d' % i, fontdict={'color': 'r'})
    k = r.trigger_kde[trigger_idx[i]]
    [scatter_ax.plot(
        [r.data[trigger_idx[i], 1], r.data[trigger_idx[i], 1] + tt[1]],
        [r.data[trigger_idx[i], 2], r.data[trigger_idx[i], 2] + tt[2]],
        'g-', lw=2.5, alpha=0.3
    ) for tt in k.data]
    ax = axes.flat[i]
    z = k.partial_marginal_pdf(xy, normed=False)
    ax.contourf(xy.toarray(0), xy.toarray(1), z, 50, cmap='afmhot')
    ax.set_aspect('equal')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.text(0, y_range[1] + .5, '%d' % i)

## BG

x_range = y_range = [-20, 20]
npt = 200
xy = DataArray.from_meshgrid(*np.meshgrid(
    np.linspace(x_range[0], x_range[1], npt),
    np.linspace(y_range[0], y_range[1], npt),
))

zbg = r.bg_kde.partial_marginal_pdf(xy, normed=False)
plt.figure()
plt.contourf(xy.toarray(0), xy.toarray(1), zbg, 50, cmap='Reds')
plt.axis('equal')
ax.set_xlim(x_range)
ax.set_ylim(y_range)

### NEW

fig = plt.figure()
scatter_ax = fig.add_subplot(111)
scatter_ax.scatter(r.data.getdim(1), r.data.getdim(2), marker='o', c='k', s=r.data.getdim(0) / max(r.data.toarray(0)) * 50, edgecolor='none', alpha=0.3)
plt.axis('equal')
xmax = max(max(r.data.toarray(1)), max(r.data.toarray(2)))
plt.axis([-xmax, xmax, -xmax, xmax])
xmax *= 1.5
scatter_ax.plot([-xmax, xmax], [0, 0], 'k--', lw=2)
scatter_ax.plot([0, 0], [-xmax, xmax], 'k--', lw=2)

# find nearest points to (10,10) and similar
coords = (
    [10, 10],
    [-10, 10],
    [10, -10],
    [-10, -10]
)
th = np.linspace(0, 2 * np.pi, 500)
t = np.linspace(0, r.max_delta_t, 500)
x = y = np.linspace(-r.max_delta_d, r.max_delta_d, 500)

x_range = y_range = [-r.max_delta_d, r.max_delta_d]
npt = 200
xy = DataArray.from_meshgrid(*np.meshgrid(
    np.linspace(x_range[0], x_range[1], npt),
    np.linspace(y_range[0], y_range[1], npt),
))

fig, surf_ax = plt.subplots(2, 2, sharex=True, sharey=True)
i = 0

for c in coords:
    idx = np.argmin(((r.data.space - c) ** 2).sumdim())
    datum = r.data.getrows(idx)
    # neighbourhood
    nn_dist = r.nn_dist[idx]
    scatter_ax.plot(nn_dist * np.cos(th) + datum[0, 1], nn_dist * np.sin(th) + datum[0, 2], 'r-', lw=2)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    k = r.trigger_kde[idx]
    ax1.plot(x, k.marginal_pdf(x, dim=1, normed=False), 'k-')
    ax1.plot(y, k.marginal_pdf(y, dim=2, normed=False), 'r-')
    q = simulate.LocalTriggeringSplitByQuartiles().default_trigger_params(*c)
    zt_th = q['intensity'] * q['time_decay'] * np.exp(-q['time_decay'] * t)
    zx_th = stats.norm.pdf(x, scale=q['sigma'][0]) * q['intensity']
    zy_th = stats.norm.pdf(y, scale=q['sigma'][1]) * q['intensity']
    ax1.plot(x, zx_th, 'k--')
    ax1.plot(y, zy_th, 'r--')
    ax2 = fig.add_subplot(122)
    ax2.plot(t, k.marginal_pdf(t, dim=0, normed=False))
    ax2.plot(t, zt_th, 'k--')

    z_local = k.partial_marginal_pdf(xy, normed=False)
    surf_ax.flat[i].contourf(xy.toarray(0), xy.toarray(1), z_local, 50, cmap='afmhot')

    i += 1