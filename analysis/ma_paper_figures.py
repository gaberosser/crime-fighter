__author__ = 'gabriel'

import cad
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib import ticker
import datetime
import numpy as np
from database import models


"""
Trend line (temporal)
"""


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


cd = {}
cd[1] = cad.CadDaily(nicl_number=1)  # violence
cd[3] = cad.CadDaily(nicl_number=3)  # burglary
cd[13] = cad.CadDaily(nicl_number=13)  # shoplifting

# setup x axis
x = [t.date() for t in cd[1].data.keys()]
y = dict(
    [(k, [len(t) for t in cd[k].data.values()]) for k in [1, 3, 13]]
)

smoothed = []

for r in y.values():
    res = moving_average(r, n=31)
    # append buffers
    smoothed.append(np.concatenate((np.nan * np.zeros(15), res, np.nan * np.zeros(15))))

start_date = datetime.date(2011, 9, 28)
end_date = start_date + datetime.timedelta(days=99)
fill_region = [True if start_date <= d <= end_date else False for d in x]

# manual tick formatting
tick_int = (5, 4, 3)

# fig = plt.figure()
fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(7.0, 4.5))
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_frame_on(False)

for i, k in enumerate([1, 3, 13]):
    # ax = fig.add_subplot(3, 1, i+1)
    ax = axs[i]
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_int[i]))
    ax.plot(x, y[k], 'k-', lw=1)
    ax.plot(x, smoothed[i], 'r-', lw=1.5, alpha=0.4)
    fig.autofmt_xdate()
    ax.grid(True, which='minor', axis='x', lw=1.5)
    ax.tick_params(which='major', axis='x', length=7, width=2, labelsize=12)
    ax.tick_params(which='major', axis='y', labelsize=12)


    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(x, 0, 1, where=fill_region, facecolor='k', alpha=0.3, transform=trans)

big_ax.set_ylabel('Daily crime count', fontsize=12)

for ax in axs:
    bbox = ax.get_position()
    ax.set_position([bbox.x0, bbox.y0, 1 - bbox.x0 - 0.02, bbox.height])

"""
Spatial distribution
"""

ma_grid = [t.mpoly for t in models.Division.objects.filter(type='monsuru_250m_grid')]

cbg = {}
for i in [1, 3, 13]:
    res, t0, cid = cad.get_crimes_by_type(nicl_type=i)
    cbg[i] = cad.CadByGrid(res, t0, data_index=cid, grid=ma_grid)

