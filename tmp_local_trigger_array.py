__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
from data.models import DataArray
from random import shuffle

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