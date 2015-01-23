__author__ = 'gabriel'

import cad
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
import datetime
import numpy as np


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

# fig = plt.figure()
fig, axs = plt.subplots(3, sharex=True, sharey=False)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
for i, k in enumerate([1, 3, 13]):
    # ax = fig.add_subplot(3, 1, i+1)
    ax = axs[i]
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.plot(x, y[k])
    ax.plot(x, smoothed[i], 'r-', lw=3, alpha=0.4)
    fig.autofmt_xdate()
    ax.grid(True, which='minor', axis='x', lw=2)
    ax.tick_params(which='major', axis='x', length=7, width=2, labelsize=22)
    ax.tick_params(which='major', axis='y', labelsize=22)


    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(x, 0, 1, where=fill_region, facecolor='k', alpha=0.3, transform=trans)

big_ax.set_ylabel('Daily crime count', fontsize=24)

# resize figure window when it appears, then run:
# plt.tight_layout()
# to maximise space usage