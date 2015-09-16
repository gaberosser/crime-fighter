__author__ = 'gabriel'

import cad
import chicago
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib import ticker
import datetime
import numpy as np
from database import models
from spacetime import DailyAggregate


def moving_average(a, n=3):
    assert n % 2 == 1, "Moving window size must be odd"
    w = (n - 1) / 2
    c = np.cumsum(a)
    f = (c[w:] - c[:-w])[:-w] / float(n)
    f = np.concatenate((np.nan * np.zeros(w), f, np.nan * np.zeros(w)))
    return f


def temporal_trend_plot(aggregate_obj, ax=None, shaded_region=None, smoothing_window=7):
    """ aggregate_obj is an instance of TemporalAggregate """
    ax = ax or plt.gca()

    # setup x axis: assume
    # x = [t.date() for t in aggregate_obj.keys()]
    x = aggregate_obj.keys()
    y = [len(t) for t in aggregate_obj.values()]
    y_smooth = moving_average(y, n=smoothing_window)

    # manual tick formatting
    # tick_int = (5, 4, 3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_int[i]))
    ax.plot(x, y, 'k-', lw=1.5)
    ax.plot(x, y_smooth, 'r-', lw=1.5, alpha=0.4)
    fig.autofmt_xdate()
    ax.grid(True, which='minor', axis='x', lw=1.5)
    ax.tick_params(which='major', axis='x', length=7, width=2, labelsize=12)
    ax.tick_params(which='major', axis='y', labelsize=12)

    if shaded_region is not None:
        fill_region = [True if shaded_region[0] <= d <= shaded_region[1] else False for d in x]
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x, 0, 1, where=fill_region, facecolor='k', alpha=0.3, transform=trans)


"""
Trend line (temporal)
"""

## Camden

start_date = datetime.datetime(2011, 3, 1)
predict_start_date = datetime.datetime(2011, 9, 28)
end_date = predict_start_date + datetime.timedelta(days=99)

nicl_types = (
    (1, 'Violence'),
    (3, 'Burglary'),
    (13, 'Shoplifting')
)

cad_data = {}

for k, v in nicl_types:
    a, b, c = cad.get_crimes_by_type(k, convert_dates=False)
    cad_data[v] = DailyAggregate(a, end_date=end_date)


fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(7.0, 4.5))
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_frame_on(False)

for i, (k, da) in enumerate(cad_data.iteritems()):
    ax = axs[i]
    temporal_trend_plot(da.data, ax=ax, shaded_region=[predict_start_date, end_date])

big_ax.set_ylabel('Daily crime count', fontsize=12)

for ax in axs:
    bbox = ax.get_position()
    ax.set_position([bbox.x0, bbox.y0, 1 - bbox.x0 - 0.02, bbox.height])


## Chicago South

chic_crime_types = (
    ('burglary', 'Burglary'),
    ('assault', 'Assault'),
    ('motor vehicle theft', 'Motor vehicle theft')
)
chic_s = chicago.get_chicago_side_polys()['South']

chic_data = {}
for k, v in chic_crime_types:
    a, b, c = chicago.get_crimes_by_type(crime_type=k,
                                         convert_dates=False,
                                         domain=chic_s,
                                         start_date=start_date,
                                         end_date=end_date)
    chic_data[v] = DailyAggregate(a, end_date=end_date)

fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(7.0, 4.5))
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_frame_on(False)

for i, (k, da) in enumerate(chic_data.iteritems()):
    ax = axs[i]
    temporal_trend_plot(da.data, ax=ax, shaded_region=[predict_start_date, end_date])

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

