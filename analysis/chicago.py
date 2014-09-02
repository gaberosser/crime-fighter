__author__ = 'gabriel'
import warnings
import collections
from django.contrib.gis.geos import Polygon, MultiPolygon, LinearRing, Point
import numpy as np
import datetime
import pytz
from database import logic, models
from point_process import estimation, models as pp_models, validate
from analysis import validation, hotspot
import settings
import os
from django.contrib.gis.gdal import DataSource
from django.db import connection
from matplotlib import pyplot as plt

CHICAGO_DATA_DIR = os.path.join(settings.DATA_DIR, 'chicago')
SRID = 2028
T0 = datetime.datetime(2001, 1, 1, 0)
Tmax = datetime.datetime(2014, 5, 24, 0)


def compute_chicago_region(fill_in=True):
    """ Get (multi) polygon representing Chicago city boundary.
        fill_in parameter specifies whether holes should be filled (better for visualisation) """
    ds = DataSource(os.path.join(CHICAGO_DATA_DIR, 'city_boundary/City_Boundary.shp'))
    mpoly = ds[0].get_geoms()[0]
    mpoly.srid = 102671
    mpoly.transform(SRID)
    if fill_in:
        mls = mpoly.boundary
        x = mls[0].coords
        x += (x[0],)
        return Polygon(x)
    return mpoly.geos


def get_crimes_by_type(crime_type='burglary', start_date=None, end_date=None, **where_strs):
    cursor = connection.cursor()
    sql = """ SELECT datetime, ST_X(location), ST_Y(location) FROM database_chicago
              WHERE LOWER(primary_type) LIKE '%{0}%' """.format(crime_type)
    if start_date:
        sql += """AND datetime >= '{0}' """.format(start_date.strftime('%Y-%m-%d'))
    if end_date:
        sql += """AND datetime <= '{0}' """.format(end_date.strftime('%Y-%m-%d'))
    for x in where_strs.values():
        sql += """AND {0}""".format(x)

    cursor.execute(sql)
    res = cursor.fetchall()
    t = [x[0] for x in res]
    t0 = min(t)
    t = [(x - t0).total_seconds() / float(60 * 60 * 24) for x in t]
    res = np.array([(t[i], res[i][1], res[i][2]) for i in range(len(res))])
    return res, t0


def pairwise_time_lag_events(max_distance=200, num_bins=None, crime_type='burglary',
                             start_date=datetime.datetime(2010, 1, 1),
                             end_date=datetime.datetime(2010, 3, 1),
                             **where_strs):
    """ Recreate Fig 1(b) Mohler et al 2011 'Self-exciting point process modeling of crime'
        max_distance is in units of metres. """

    res, t0 = get_crimes_by_type(crime_type=crime_type, start_date=start_date, end_date=end_date, **where_strs)
    res = res[np.argsort(res[:, 0])]

    # use SEPP model linkage method
    sepp = pp_models.PointProcess(max_trigger_d=max_distance, max_trigger_t=1e12)
    sepp.set_data(res)
    sepp.set_linkages()
    ii, jj = sepp.linkage

    dt = [res[j, 0] - res[i, 0] for i, j in zip(ii, jj)]
    max_dt = max(dt)
    k = 0.5 * max_dt * (max_dt - 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dt, num_bins or range(int(max_dt) + 1), normed=True, edgecolor='none', facecolor='gray')
    ax.plot(np.arange(1, max_dt), np.arange(1, max_dt)[::-1] / k, 'k--')
    ax.set_xlabel('Time difference (days)')
    ax.set_ylabel('Event pair density')
    plt.show()

    return dt


def apply_point_process():

    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        start_date=datetime.datetime(2010, 3, 1, 0),
        end_date=datetime.datetime(2010, 6, 1, 0)
    )

    max_trigger_t = 40 # units days
    max_trigger_d = 300 # units metres
    min_bandwidth = np.array([0.3, 5., 5.])
    # min_bandwidth = None

    # manually estimate p initially
    est = lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02)
    r = pp_models.PointProcessStochasticNn(estimator=est, max_trigger_t=max_trigger_t, max_trigger_d=max_trigger_d,
                            min_bandwidth=min_bandwidth)
    # r = pp_models.PointProcessStochastic(estimator=est, max_trigger_t=max_trigger_t, max_trigger_d=max_trigger_d,
    #                         min_bandwidth=[1., 5., 5.])
    ps = r.train(data=res, niter=12, tol_p=5e-5)
    return r, ps


def validate_point_process(
        start_date=datetime.datetime(2001, 3, 1, 0),
        training_size=None,
        num_validation=20,
        num_pp_iter=20,
        grid=500,
        prediction_dt=1):

    if not training_size:
        # use all possible days before start_date as training
        training_size = (start_date - T0).days
    else:
        if (start_date - T0).days < training_size:
            new_training_size = (start_date - T0).days
            warnings.warn("Requested %d training days, but only have data for %d days." % (training_size, new_training_size))
            training_size = new_training_size

    # compute number of days in date range
    pre_start_date = start_date - datetime.timedelta(days=training_size)
    ndays = training_size + num_validation
    end_date = start_date + datetime.timedelta(days=num_validation)

    poly = compute_chicago_region()
    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        start_date=pre_start_date,
        end_date=end_date
    )
    vb = validate.PpValidation(res, spatial_domain=poly, model_kwargs={
        'max_trigger_t': 30,
        'max_trigger_d': 200,
        'estimator': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02),
    })
    vb.set_grid(grid)
    vb.set_t_cutoff(training_size, b_train=False)

    res = vb.run(time_step=prediction_dt, t_upper=training_size + ndays,
                 train_kwargs={'niter': num_pp_iter, 'tol_p': 1e-5},
                 verbose=True)

    return res, vb


def validate_point_process_multi(
        start_date=datetime.datetime(2001, 6, 1, 0),
        training_size=90,
        num_validation=20,
        num_pp_iter=20,
        grid_size=500,
        dt=180,
        callback_func=None):

    """ Validate point process model over the Chicago dataset.
:param start_date: Initial date from which data are used, *must include training data too*
:param training_size: Number of days to use for model training each time.  Prevents blow-up in memory requirements.
:param num_validation: Number of steps in time to use for each time slice considered
:param num_pp_iter: Number of iterations of SEPP per training run
:param grid_size: Length of grid square
:param dt: Step in time - we slice the entire dataset with this stride length
:param callback_func: Optional function to call with the data from each successful validation block (time slice).
May be useful for pickling output incrementally.
:return: res is a list of the output from validation.run() calls, pps is a list of the resulting trained SEPP model,
            taken from the final run of each time slice.
"""

    grid = grid_size

    # validate the SEPP method over the whole dataset, running 20 iterations at a time and
    # applying to different temporal slices

    res = collections.OrderedDict()
    t = start_date

    while t < Tmax:
        try:
            res[t], vb = validate_point_process(
                start_date=t,
                training_size=training_size,
                num_validation=num_validation,
                num_pp_iter=num_pp_iter,
                grid=grid
            )
            grid = vb.roc
        except Exception as exc:
            # TODO: something smarter
            print repr(exc)
        if callback_func:
            callback_func({t: res[t]})
        t += datetime.timedelta(days=dt)

    return res


    # pre_start_date = start_date - datetime.timedelta(days=training_size)
    #
    # poly = compute_chicago_region()
    # data, t0 = get_crimes_by_type(
    #     crime_type='burglary',
    #     datetime__gte=pre_start_date
    # )
    # nslices = int((max(data[:, 0]) - training_size - num_validation) / dt)
    # res = []
    # pps = []
    #
    # for n in range(nslices):
    #     tn = dt * n
    #     # slice data to avoid training on all past data
    #     this_data = data[data[:, 0] >= tn]
    #     vb = validate.PpValidation(this_data, spatial_domain=poly, model_kwargs={
    #         'max_trigger_t': 30,
    #         'max_trigger_d': 200,
    #         'estimator': lambda x: estimation.initial_guess_educated(x, ct=1, cd=0.02),
    #     })
    #     vb.set_grid(grid_size)
    #     vb.set_t_cutoff(training_size + tn)
    #     res.append(vb.run(dt=1, t_upper=training_size + num_validation + tn, niter=num_pp_iter))
    #     pps.append(vb.model)
    #
    # return res, pps


def validate_historic_kernel(start_date=datetime.datetime(2001, 3, 1, 0),
                           grid_size=250,
                           previous_n_days=7,
                           n_iter=20):

    pre_start_date = start_date - datetime.timedelta(days=previous_n_days)
    end_date = start_date + datetime.timedelta(days=n_iter)

    poly = compute_chicago_region()
    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        start_date=pre_start_date,
        end_date=end_date
    )

    # use basic historic data spatial hotspot
    sk = hotspot.SKernelHistoric(previous_n_days)
    vb = validation.ValidationBase(res, hotspot.Hotspot, poly, model_args=(sk,))
    vb.set_grid(grid_length=grid_size)
    vb.set_t_cutoff(previous_n_days)
    ranks, carea, cfrac, pai = vb.run(dt=1, t_upper=previous_n_days + n_iter)

    return vb.grid, ranks, carea, cfrac, pai


def validate_historic_kernel_multi(start_date=datetime.datetime(2001, 3, 1, 0),
                                   grid_size=500,
                                   previous_n_days=7,
                                   n_iter=20,
                                   dt=180):
    # validate the historic kernel method over the whole dataset, running 20 iterations at a time and
    # applying to different temporal slices
    pre_start_date = start_date - datetime.timedelta(days=previous_n_days)

    poly = compute_chicago_region()
    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        datetime__gte=pre_start_date
    )
    nslices = int(max(res[:, 0]) / dt)
    sk = hotspot.SKernelHistoric(previous_n_days)
    vb = validation.ValidationBase(res, hotspot.Hotspot, poly, model_args=(sk,))
    vb.set_grid(grid_length=grid_size)
    vb.set_t_cutoff(previous_n_days)

    res = []

    for n in range(nslices):
        res.append(vb.run(dt=1, t_upper=previous_n_days + n_iter + n * dt))
        vb.set_t_cutoff(previous_n_days + (n + 1) * dt)

    return res


if __name__ == '__main__':

    start_date=datetime.datetime(2001, 1, 1, 0)
    end_date=datetime.datetime(2003, 1, 1)
    first_training_size = 50

    poly = compute_chicago_region()
    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        start_date=start_date,
        end_date=end_date
    )

    vb = validate.PpValidation(res, spatial_domain=poly, model_kwargs={
        'max_trigger_t': 30,
        'max_trigger_d': 200,
        'estimator': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02),
    })
    vb.set_grid(250)
    vb.set_t_cutoff(first_training_size, b_train=False)

    sepp_res = vb.run(time_step=1, t_upper=first_training_size + 1,
                 train_kwargs={'niter': 20, 'tol_p': 1e-5},
                 verbose=True)

    # use basic historic data spatial hotspot
    sk = hotspot.SKernelHistoric(first_training_size) # use heatmap from same period
    vb_sk = validation.ValidationBase(res, hotspot.Hotspot, poly, model_args=(sk,))
    vb_sk.roc.copy_grid(vb.roc)
    # vb_sk._grid = vb._grid
    # vb_sk.centroids = vb.centroids
    # vb_sk.a = vb.a
    vb_sk.set_t_cutoff(first_training_size, b_train=False)
    sk_res = vb_sk.run(time_step=1, t_upper=first_training_size + 1,
                         verbose=True)

    # use variable bandwidth KDE
    skvb = hotspot.SKernelHistoricVariableBandwidthNn(first_training_size)
    vb_skvb = validation.ValidationBase(res, hotspot.Hotspot, poly, model_args=(skvb,))
    vb_skvb.roc.copy_grid(vb.roc)
    # vb_skvb._grid = vb._grid
    # vb_skvb.centroids = vb.centroids
    # vb_skvb.a = vb.a
    vb_skvb.set_t_cutoff(first_training_size, b_train=False)
    skvb_res = vb_skvb.run(time_step=1, t_upper=first_training_size + 1,
                         verbose=True)
