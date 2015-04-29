__author__ = 'gabriel'

from point_process import models, estimation, simulate, plotting
from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
from data.models import DataArray, CartesianSpaceTimeData
from copy import deepcopy
import logging
import dill

f = open('tmp_store.pickle', 'r')
d = dill.load(f)

max_delta_t = 200
max_delta_d = 3.

bg_kde_kwargs = {
'number_nn': [100, 15],
'strict': False,
}

trigger_kde_kwargs = {
'number_nn': 15,
'strict': False,
}

r = models.SeppStochasticNn(data=d['data'], max_delta_d=max_delta_d, max_delta_t=max_delta_t,bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
r.p = d['p']
r.set_kdes()

x_range = y_range = [-20, 20]
npt = 200
xy = CartesianSpaceTimeData.from_meshgrid(*np.meshgrid(np.linspace(x_range[0], x_range[1], npt), np.linspace(y_range[0], y_range[1], npt)))

xyt = CartesianSpaceTimeData(np.ones(npt**2) * 100)  # value of t should be irrelevant
xyt = xyt.adddim(xy)
xyt.original_shape = (npt, npt)

bg = r.background_density(xyt, spatial_only=True)
trigger = r.trigger_density_in_place(xyt, spatial_only=True)

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.contourf(xy.toarray(0), xy.toarray(1), bg, 50, cmap='afmhot_r')
ax2.contourf(xy.toarray(0), xy.toarray(1), trigger, 50, cmap='afmhot_r')
ax3.contourf(xy.toarray(0), xy.toarray(1), trigger / (trigger + bg), 50, cmap='afmhot_r')