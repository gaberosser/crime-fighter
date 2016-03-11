from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib import ticker
import datetime
import numpy as np
import bisect


def moving_average(a, n=3):
    assert n % 2 == 1, "Moving window size must be odd"
    w = (n - 1) / 2
    f = np.convolve(a, np.ones(n), 'valid') / float(n)
    f = np.concatenate((np.nan * np.zeros(w), f, np.nan * np.zeros(w)))
    return f


def temporal_trend_plot_one(aggregate_obj, ax=None, shaded_region=None, smoothing_window=7):
    """ aggregate_obj is an instance of TemporalAggregate """
    ax = ax or plt.gca()
    fig = ax.get_figure()

    # setup x axis: assume
    # x = [t.date() for t in aggregate_obj.keys()]
    x = aggregate_obj.keys()
    y = [len(t) for t in aggregate_obj.values()]
    y_smooth = moving_average(y, n=smoothing_window)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.plot(x, y, 'k-', lw=1.5)
    ax.plot(x, y_smooth, 'r-', lw=2.5, alpha=0.75)
    fig.autofmt_xdate()
    ax.grid(True, which='minor', axis='x', lw=1.5)
    ax.tick_params(which='major', axis='x', length=7, width=2, labelsize=12)
    ax.tick_params(which='major', axis='y', labelsize=12)

    if shaded_region is not None:
        fill_region = [True if shaded_region[0] <= d <= shaded_region[1] else False for d in x]
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x, 0, 1, where=fill_region, facecolor='k', alpha=0.3, transform=trans)
        

def accuracy_over_time(vb_res, coverage=(0.01, 0.05, 0.1, 0.2)):
    if not hasattr(coverage, '__iter__'):
        coverage = [coverage]
    pai = []
    hr = []
    for i in range(len(vb_res['cumulative_area'])):
        x = vb_res['cumulative_area'][i]
        this_hr = []
        this_pai = []
        for c in coverage:
            idx = bisect.bisect_left(x, c)
            this_hr.append(vb_res['cumulative_crime'][i][idx])
            this_pai.append(vb_res['pai'][i][idx])
        hr.append(this_hr)
        pai.append(this_pai)
    return np.array(hr), np.array(pai)
