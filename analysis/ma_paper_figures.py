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
from spacetime import DailyAggregate, SpatialAggregate
from plotting import spatial
from analysis.spatial import create_spatial_grid


start_date = datetime.datetime(2011, 3, 1)
predict_start_date = datetime.datetime(2011, 9, 28)
end_date = predict_start_date + datetime.timedelta(days=99)


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


def temporal_trend_figure(data):

    left_buffer = 0.12
    fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(9.0, 4.5))
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_frame_on(False)
    big_ax.set_position([0., 0., 1. - 2 * left_buffer, 1.])

    for i, (k, da) in enumerate(data.iteritems()):
        ax = axs[i]
        da.aggregate()
        temporal_trend_plot_one(da.data, ax=ax, shaded_region=[predict_start_date, end_date])
        # add crime type text
        text_y = np.mean(ax.get_ylim())
        ax.text(datetime.datetime(2012, 1, 7), text_y, k, fontsize=12)

    big_ax.set_ylabel('Daily crime count', fontsize=12)

    for ax in axs:
        bbox = ax.get_position()
        ax.set_position([left_buffer, bbox.y0, 1 - 2 * left_buffer, bbox.height])


def spatial_distribution_one(aggregate_data, ax=None, domain=None, scale_bar_loc='se'):
    ax = ax or plt.gca()
    y = [len(t) for t in aggregate_data.values()]
    spatial.plot_shaded_regions(aggregate_data.keys(),
                                y,
                                ax=ax,
                                cmap=plt.get_cmap('autumn_r'),
                                vmin=1e-6,
                                domain=domain,
                                scale_bar_loc=scale_bar_loc)


if __name__ == "__main__":

    chic_s_domain = chicago.get_chicago_side_polys(as_shapely=True)['South']
    camden_domain = cad.get_camden_region(as_shapely=True)
    chic_s_grid = create_spatial_grid(chic_s_domain, 250)[0]
    cad_grid = [t.mpoly for t in models.Division.objects.filter(type='monsuru_250m_grid')]

    ## Camden

    nicl_types = (
        (1, 'Violence'),
        (3, 'Burglary'),
        (13, 'Shoplifting')
    )

    cad_data = {}
    cad_spatial_data = {}

    for k, v in nicl_types:
        a, b, c = cad.get_crimes_by_type(k, convert_dates=False)
        cad_data[v] = DailyAggregate(a, end_date=end_date)

        sa = SpatialAggregate(a)
        sa.set_areal_units(cad_grid)
        sa.aggregate()
        cad_spatial_data[v] = sa

    temporal_trend_figure(cad_data)
    plt.gcf().savefig('camden_time_series.png')
    plt.gcf().savefig('camden_time_series.pdf')

    ## Chicago South

    chic_crime_types = (
        ('burglary', 'Burglary'),
        ('assault', 'Assault'),
        ('motor vehicle theft', 'Motor vehicle\ntheft')
    )


    chic_data = {}
    chic_spatial_data = {}

    for k, v in chic_crime_types:
        a, b, c = chicago.get_crimes_by_type(crime_type=k,
                                             convert_dates=False,
                                             domain=chic_s_domain,
                                             start_date=start_date,
                                             end_date=end_date)
        chic_data[v] = DailyAggregate(a, end_date=end_date)

        sa = SpatialAggregate(a, domain=chic_s_domain)
        sa.set_areal_units(chic_s_grid)
        sa.aggregate()
        chic_spatial_data[v] = sa

    temporal_trend_figure(chic_data)
    plt.gcf().savefig('chicago_south_time_series.png')
    plt.gcf().savefig('chicago_south_time_series.pdf')


    # Spatial distribution plots


    for i, (k, sa) in enumerate(cad_spatial_data.iteritems()):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        spatial_distribution_one(sa.data, ax=ax, domain=camden_domain)
        fig.savefig('camden_spatial_%s.png' % k.lower().replace(' ', '_'))
        fig.savefig('camden_spatial_%s.pdf' % k.lower().replace(' ', '_'))

    for i, (k, sa) in enumerate(chic_spatial_data.iteritems()):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        spatial_distribution_one(sa.data, ax=ax, domain=chic_s_domain, scale_bar_loc='sw')
        fig.savefig('chicago_south_spatial_%s.png' % k.lower().replace(' ', '_'))
        fig.savefig('chicago_south_spatial_%s.pdf' % k.lower().replace(' ', '_'))