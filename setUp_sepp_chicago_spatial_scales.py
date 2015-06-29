__author__ = 'gabriel'
from analysis import chicago
from database import models
from point_process import plotting as pp_plotting
import numpy as np
from matplotlib import pyplot as plt

chic = chicago.compute_chicago_region(fill_in=True)
southwest = models.ChicagoDivision.objects.get(name='Southwest').mpoly
south = models.ChicagoDivision.objects.get(name='South').mpoly
west = models.ChicagoDivision.objects.get(name='West').mpoly

southwest_buf = southwest.buffer(1500)
south_buf = south.buffer(1500)
west_buf = west.buffer(1500)

# res_chic, vb_chic = chicago.validate_point_process()

r_chic, ps_chic = chicago.apply_point_process()
r_sw, ps_sw = chicago.apply_point_process(domain=southwest_buf)
r_s, ps_s = chicago.apply_point_process(domain=south_buf)
r_w, ps_w = chicago.apply_point_process(domain=west_buf)

t = r_chic.data[-1, 0]
dx = 20

# heatmap of BG: whole region and subregions
fig = plt.figure()
ax = fig.add_subplot(111)
xyz = pp_plotting.prediction_heatmap(r_chic, t, poly=chic, ax=ax, dx=dx, fmax=0.98)
vmax = np.max(xyz[2])

fig = plt.figure()
ax = fig.add_subplot(111)
pp_plotting.prediction_heatmap(r_sw, t, poly=southwest, ax=ax, dx=dx, vmax=vmax)

fig = plt.figure()
ax = fig.add_subplot(111)
pp_plotting.prediction_heatmap(r_s, t, poly=south, ax=ax, dx=dx, vmax=vmax)

fig = plt.figure()
ax = fig.add_subplot(111)
pp_plotting.prediction_heatmap(r_w, t, poly=west, ax=ax, dx=dx, vmax=vmax)