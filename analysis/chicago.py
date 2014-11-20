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
import dill

CHICAGO_DATA_DIR = os.path.join(settings.DATA_DIR, 'chicago')
SRID = 2028
T0 = datetime.datetime(2001, 1, 1, 0)
Tmax = datetime.datetime(2014, 5, 24, 0)


def compute_chicago_region(fill_in=True):
    """ Get (multi) polygon representing Chicago city boundary.
        fill_in parameter specifies whether holes should be filled (better for visualisation) """
    mpoly = models.ChicagoDivision.objects.get(name='Chicago').mpoly
    if fill_in:
        mls = mpoly.boundary
        x = mls[0].coords
        x += (x[0],)
        return Polygon(x)
    return mpoly


def compute_chicago_marine_section():
    mpoly = compute_chicago_region()
    xmin, ymin, xmax, ymax = mpoly.extent
    a = [(x, y) for (x, y) in mpoly.coords[0] if x == xmax]
    b = [(x, y) for (x, y) in mpoly.coords[0] if y == ymax]
    new_poly = Polygon([
        (b[0][0], ymax),
        (xmax, ymax),
        (xmax, a[0][1]),
        (b[0][0], a[0][1]),
        (b[0][0], ymax),
    ])
    return new_poly.difference(mpoly)


def compute_chicago_land_buffer():
    mpoly = compute_chicago_region()
    xmin, ymin, xmax, ymax = mpoly.extent
    a = [(x, y) for (x, y) in mpoly.coords[0] if x == xmax]
    b = [(x, y) for (x, y) in mpoly.coords[0] if y == ymax]
    new_poly = Polygon([
        (b[0][0], ymax),
        (xmin, ymax),
        (xmin, ymin),
        (xmax, ymin),
        mpoly.centroid.coords,
        (b[0][0], ymax),
    ])
    return new_poly.difference(mpoly)


def get_crimes_by_type(crime_type='burglary', start_date=None, end_date=None, domain=None, **where_strs):
    """
    Get all matching crimes from the Chicago dataset
    :param crime_type:
    :param start_date:
    :param end_date:
    :param domain: geos.GEOSGeometry object
    :param where_strs:
    :return:
    """
    cursor = connection.cursor()
    sql = """ SELECT datetime, ST_X(location), ST_Y(location) FROM database_chicago
              WHERE LOWER(primary_type) LIKE '%{0}%' """.format(crime_type.lower())
    if start_date:
        sql += """AND datetime >= '{0}' """.format(start_date.strftime('%Y-%m-%d'))
    if end_date:
        sql += """AND datetime <= '{0}' """.format(end_date.strftime('%Y-%m-%d'))
    if domain:
        sql += """AND ST_Intersects(location, ST_GeomFromText('{0}', {1}))"""\
            .format(domain.wkt, SRID)
    for x in where_strs.values():
        sql += """AND {0}""".format(x)

    cursor.execute(sql)
    res = cursor.fetchall()
    t = [x[0] for x in res]
    t0 = min(t)
    t = [(x - t0).total_seconds() / float(60 * 60 * 24) for x in t]
    res = np.array([(t[i], res[i][1], res[i][2]) for i in range(len(res))])
    res = res[np.argsort(res[:, 0])]
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


def apply_point_process(start_date=datetime.datetime(2010, 3, 1, 0),
                        end_date=datetime.datetime(2010, 6, 1, 0),
                        domain=None,
                        niter=15,
                        min_bandwidth=None,
                        max_delta_t=40,
                        max_delta_d=300,
                        num_nn=None,
                        estimate_kwargs=None,
                        pp_class=pp_models.SeppStochasticNnReflected,
                        tol_p=None):

    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        start_date=start_date,
        end_date=end_date,
        domain=domain
    )



    if min_bandwidth is None:
        min_bandwidth = np.array([0.3, 5., 5.])

    if num_nn is not None:
        if len(num_nn) != 2:
            raise AttributeError("Must supply two num_nn values: [1D case, 2/3D case]")
        num_nn_bg = num_nn
        num_nn_trig = num_nn[1]
    else:
        num_nn_bg = [101, 16]
        num_nn_trig = 15

    bg_kde_kwargs = {
        'number_nn': num_nn_bg,
    }

    trigger_kde_kwargs = {
        'min_bandwidth': min_bandwidth,
        'number_nn': num_nn_trig,
    }

    if not estimate_kwargs:
        estimate_kwargs = {
            'ct': 1,
            'cd': 0.02
        }

    r = pp_class(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                 bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)

    # r = pp_models.SeppStochasticNn(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = pp_models.SeppStochasticNnReflected(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = pp_models.SeppStochasticNnOneSided(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    p = estimation.estimator_bowers(res, r.linkage, **estimate_kwargs)
    r.p = p

    # train on ALL data
    ps = r.train(niter=niter, tol_p=tol_p)
    return r, ps


def validate_point_process(
        start_date=datetime.datetime(2012, 3, 1, 0),
        training_size=365,
        num_validation=20,
        num_pp_iter=20,
        grid=500,
        prediction_dt=1, true_dt_plus=1,
        domain=None,
        model_kwargs=None):

    if not model_kwargs:
        model_kwargs = {
            'max_delta_t': 30,
            'max_delta_d': 200,
            'bg_kde_kwargs': {'number_nn': [101, 16]},
            'trigger_kde_kwargs': {'number_nn': 15,
                                   'min_bandwidth': [0.3, 5, 5]}
        }

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

    if domain is None:
        domain = compute_chicago_region()

    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        start_date=pre_start_date,
        end_date=end_date,
        domain=domain,
    )
    vb = validate.SeppValidation(res, spatial_domain=domain, **model_kwargs)
    vb.set_grid(grid)
    vb.set_t_cutoff(training_size, b_train=False)

    ## TODO: check the number of iterations reported is as expected here
    res = vb.run(time_step=prediction_dt, t_upper=ndays, true_dt_plus=true_dt_plus,
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


# implementation script
# test the effect of delta_t and delta_d on the output
def implement_delta_effect(outfile):

    dts = [10, 20, 30, 40, 50, 60, 90]
    dds = [20, 50, 100, 200, 300, 500, 1000, 5000]

    start_date = datetime.datetime(2010, 1, 1, 0)
    end_date = datetime.datetime(2010, 7, 1, 0)
    niter = 15

    dt_grid, dd_grid = np.meshgrid(dts, dds)

    res = []

    with open(outfile, 'w') as f:
        for dt, dd in zip(dt_grid.flat, dd_grid.flat):
            try:
                r, p = apply_point_process(
                    start_date=start_date,
                    end_date=end_date,
                    niter=niter,
                    max_delta_t=dt,
                    max_delta_d=dd,
                )
                res.append((dt, dd, r))

                dill.dump((dt, dd, r), f)

            except Exception:
                pass

    return res


# implementation script
# test the effect of spatial and temporal domain on the triggering function
def implement_spatial_temporal_domain_effect():

    niter = 35

    start_date_1 = datetime.datetime(2010, 1, 1, 0)
    start_date_2 = datetime.datetime(2010, 1, 7, 0)
    end_date_1 = datetime.datetime(2010, 7, 1, 0)
    end_date_2 = datetime.datetime(2011, 1, 1, 0)

    centroid_1 = Point((445250, 4628000))
    domain_m = centroid_1.buffer(5000)
    domain_l = centroid_1.buffer(8000)

    centroid_2 = Point((448000, 4621500))
    domain_2 = centroid_2.buffer(5000)

    res = {}

    # spatial translation
    res['base'], ps = apply_point_process(start_date=start_date_1,
                                         end_date=end_date_1,
                                         domain=domain_m,
                                         niter=niter)

    res['spatial_translation'], ps = apply_point_process(start_date=start_date_1,
                                                         end_date=end_date_1,
                                                         domain=domain_2,
                                                         niter=niter)

    # spatial enlargement

    res['spatial_enlargement'], ps = apply_point_process(start_date=start_date_1,
                                                         end_date=end_date_1,
                                                         domain=domain_l,
                                                         niter=niter)

    # temporal translation

    res['temporal_translation'], ps = apply_point_process(start_date=start_date_2,
                                                          end_date=end_date_2,
                                                          domain=domain_m,
                                                          niter=niter)

    # temporal enlargement

    res['temporal_enlargement'], ps = apply_point_process(start_date=start_date_1,
                                                          end_date=end_date_2,
                                                          domain=domain_m,
                                                          niter=niter)

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
