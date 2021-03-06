from validation import validation, hotspot

__author__ = 'gabriel'
import warnings
import collections
from django.contrib.gis.geos import Polygon, Point
import numpy as np
import datetime
from time import time
from database import models
from point_process import estimation, models as pp_models, validate
from analysis import spatial
import settings
import os
from django.db import connection
from matplotlib import pyplot as plt
from plotting.spatial import plot_shapely_geos, geodjango_to_shapely
import dill
import copy

CHICAGO_DATA_DIR = os.path.join(settings.DATA_DIR, 'chicago')
SRID = 2028
T0 = datetime.datetime(2001, 1, 1, 0)
Tmax = datetime.datetime(2014, 5, 24, 0)


# TODO
# def stik(poly, start_date, end_date)


def compute_chicago_region(fill_in=True, as_shapely=False):
    """ Get (multi) polygon representing Chicago city boundary.
        fill_in parameter specifies whether holes should be filled (better for visualisation) """
    mpoly = models.ChicagoDivision.objects.get(name='Chicago').mpoly
    if fill_in:
        mls = mpoly.boundary
        x = mls[0].coords
        x += (x[0],)
        if as_shapely:
            return spatial.geodjango_to_shapely(Polygon(x))
        else:
            return Polygon(x)
    if as_shapely:
        return spatial.geodjango_to_shapely(mpoly)
    else:
        return mpoly


def get_chicago_side_polys(as_shapely=True):
    sides = models.ChicagoDivision.objects.filter(type='chicago_side')
    res = {}
    for s in sides:
        if as_shapely:
            res[s.name] = spatial.geodjango_to_shapely(s.mpoly.simplify())
        else:
            res[s.name] = s.mpoly.simplify()
    return res


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





def get_crimes_by_type(crime_type='burglary',
                       start_date=None,
                       end_date=None,
                       domain=None,
                       convert_dates=True,
                       **where_strs):
    """
    Get all matching crimes from the Chicago dataset
    :param crime_type: String or an iterable of strings corresponding to the primary_type field (case insensitive).
    :param start_date:
    :param end_date:
    :param domain: geos.GEOSGeometry object
    :param convert_dates: If True, dates are converted to the number of days after t0 (float)
    :param where_strs:
    :return:
    """
    if start_date and not isinstance(start_date, datetime.datetime):
        # start_date refers to 00:00 onwards
        start_date = datetime.datetime.combine(start_date, datetime.time(0))
    if end_date and not isinstance(end_date, datetime.datetime):
        # end_date refers to 23:59:59 backwards
        end_date = end_date + datetime.timedelta(days=1)
        end_date = datetime.datetime.combine(end_date, datetime.time(0)) - datetime.timedelta(seconds=1)

    cursor = connection.cursor()
    sql = """ SELECT "number", datetime, ST_X(location), ST_Y(location) FROM database_chicago """
    if hasattr(crime_type, '__iter__'):
        sql += """WHERE LOWER(primary_type) IN ({0}) """.format(','.join(["'%s'" % t.lower() for t in crime_type]))
    else:
        sql += """WHERE LOWER(primary_type) LIKE '%{0}%' """.format(crime_type.lower())
    if start_date:
        sql += """AND datetime >= '{0}' """.format(start_date.strftime('%Y-%m-%d %H:%M:%S'))
    if end_date:
        sql += """AND datetime <= '{0}' """.format(end_date.strftime('%Y-%m-%d %H:%M:%S'))
    if domain:
        sql += """AND ST_Intersects(location, ST_GeomFromText('{0}', {1}))"""\
            .format(domain.wkt, SRID)
    for x in where_strs.values():
        sql += """AND {0}""".format(x)

    cursor.execute(sql)
    res = cursor.fetchall()
    cid = np.array([x[0] for x in res])
    t0 = min([x[1] for x in res])
    xy = np.array([(res[i][2], res[i][3]) for i in range(len(res))])
    if convert_dates:
        t = np.array([[(x[1] - t0).total_seconds() / float(60 * 60 * 24)] for x in res])
    else:
        t = np.array([[x[1]] for x in res])
    # res = np.array([(t[i], res[i][2], res[i][3]) for i in range(len(res))])
    res = np.hstack((t, xy))

    # sort data
    sort_idx = np.argsort(res[:, 0])
    res = res[sort_idx]
    cid = cid[sort_idx]

    return res, t0, cid


def crime_numbers_by_side(crime_types=('burglary', 'assault'),
                            start_date=datetime.date(2011, 3, 1),
                            end_date=datetime.date(2012, 3, 1)):
    domains = get_chicago_side_polys(as_shapely=True)
    ndata = collections.defaultdict(dict)
    for ct in crime_types:
        n = []
        for k in domains:
            domain = domains[k]
            data, _, _ = get_crimes_by_type(ct, start_date=start_date, end_date=end_date, domain=domain)
            n.append(len(data))
        ndata[ct] = np.array(n)

    return ndata, domains.keys()


def crime_numbers_by_side_bar(ndata, domain_labels, buffer=0.05):
    colour_cycle = ('b', 'k', 'r', 'g', 'y')
    crime_types = ndata.keys()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, ct in enumerate(crime_types):
        c = colour_cycle[np.mod(i, len(colour_cycle))]
        x = np.arange(len(domain_labels)) + buffer + (1. - 2 * buffer) * float(i)/len(crime_types)
        ax.bar(x, ndata[ct], width=(1. - 2 * buffer)/len(crime_types), facecolor=c, edgecolor=c)
    ax.set_xticks(np.arange(len(domain_labels)) + 0.5)
    ax.set_xticklabels(domain_labels, rotation=45)
    ax.set_ylabel('Crime count')
    ax.legend(crime_types)


def crime_density_by_side_bar(ndata, domain_labels, buffer=0.05):
    domains = get_chicago_side_polys(as_shapely=True)
    a = np.array([domains[k].area for k in domain_labels]) / 1e6  # km ^ 2
    colour_cycle = ('b', 'k', 'r', 'g', 'y')
    crime_types = ndata.keys()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    max_y = 0
    for i, ct in enumerate(crime_types):
        c = colour_cycle[np.mod(i, len(colour_cycle))]
        x = np.arange(len(domain_labels)) + buffer + (1. - 2 * buffer) * float(i)/len(crime_types)
        y = ndata[ct] / a
        max_y = max(max_y, y.max())
        ax.bar(x, y, width=(1. - 2 * buffer)/len(crime_types), facecolor=c, edgecolor=c)
    ax.set_xticks(np.arange(len(domain_labels)) + 0.5)
    ax.set_xticklabels(domain_labels, rotation=45)
    ax.set_ylabel(r'Crimes km$^{-2}$')
    ax.legend(crime_types)
    ax.set_ylim([0, max_y * 1.4])  # leave some space for the legend
    plt.tight_layout()


def spatial_repeat_analysis(crime_type='burglary', domain=None, plot_osm=False, **kwargs):
    data, t0, cid = get_crimes_by_type(crime_type=crime_type, domain=domain, **kwargs)
    xy = data[:, 1:]
    rpt = collections.defaultdict(int)
    uniq = collections.defaultdict(int)
    for t in xy:
        if np.sum((np.sum(xy == t, axis=1) == 2)) > 1:
            rpt[tuple(t)] += 1
        else:
            uniq[tuple(t)] += 1

    # plotting
    domain = domain or compute_chicago_region()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # outline
    plot_shapely_geos(domain, ax=ax, fc='None')

    # off-grid repeat locations
    tt = np.array(rpt.keys())
    ax.plot(tt[:, 0], tt[:, 1], 'ok')

    #
    tt = np.array(uniq.keys())
    ax.plot(tt[:, 0], tt[:, 1], 'o', color='#CCCCCC', alpha=0.6)

    # x_max, y_max = np.max(np.array(camden.mpoly[0].coords[0]), axis=0)
    # x_min, y_min = np.min(np.array(camden.mpoly[0].coords[0]), axis=0)

    # ax.set_xlim(np.array([-150, 150]) + np.array([x_min, x_max]))
    # ax.set_ylim(np.array([-150, 150]) + np.array([y_min, y_max]))
    ax.set_aspect('equal')
    ax.axis('off')

    plt.draw()

    return rpt, uniq


def pairwise_time_lag_events(max_distance=200, num_bins=None, crime_type='burglary',
                             start_date=datetime.datetime(2010, 1, 1),
                             end_date=datetime.datetime(2010, 3, 1),
                             **where_strs):
    """ Recreate Fig 1(b) Mohler et al 2011 'Self-exciting point process modeling of crime'
        max_distance is in units of metres. """

    res, t0, cid = get_crimes_by_type(crime_type=crime_type, start_date=start_date, end_date=end_date, **where_strs)
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


def plot_domain(city_domain, sub_domain, ax=None, set_axis=True):
    bbox = np.array(city_domain.buffer(100).bounds)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    plot_shapely_geos(city_domain, ax=ax)
    plot_shapely_geos(sub_domain, ax=ax, facecolor='k')
    if set_axis:
        plt.axis('equal')
        plt.axis(np.array(bbox)[(0, 2, 1, 3),])
        plt.axis('off')


def plot_domains(name=None, ax=None, set_axis=True):
    if name is None and ax is not None:
        raise AttributeError("If no single domain is specified, we have to make new figures for each plot")
    city = geodjango_to_shapely(
        models.ChicagoDivision.objects.get(name='ChicagoFilled').mpoly.simplify(0)
    )
    domains = get_chicago_side_polys(as_shapely=True)
    if name is not None:
        plot_domain(city, domains[name], ax=ax, set_axis=set_axis)
    else:
        for k, v in domains.items():
            plot_domain(city, v)



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
                        seed=42,
                        remove_coincident_pairs=False):

    print "Getting data..."
    res, t0, cid = get_crimes_by_type(
        crime_type='burglary',
        start_date=start_date,
        end_date=end_date,
        domain=domain
    )



    # if min_bandwidth is None:
    #     min_bandwidth = np.array([0.3, 5., 5.])

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
        'strict': False,
    }

    trigger_kde_kwargs = {
        'min_bandwidth': min_bandwidth,
        'number_nn': num_nn_trig,
        'strict': False,
    }

    if not estimate_kwargs:
        estimate_kwargs = {
            'ct': 0.1,
            'cd': 50,
            'frac_bg': None
        }

    print "Instantiating SEPP class..."
    r = pp_class(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                 bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs,
                 remove_coincident_pairs=remove_coincident_pairs)

    # r = pp_models.SeppStochasticNn(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = pp_models.SeppStochasticNnReflected(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = pp_models.SeppStochasticNnOneSided(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)

    print "Computing initial probability matrix estimate..."
    p = estimation.estimator_exp_gaussian(res, r.linkage, **estimate_kwargs)
    r.p = p

    # train on ALL data
    if seed:
        r.set_seed(seed)

    print "Starting training..."
    ps = r.train(niter=niter)
    return r


def validate_point_process(
        start_date=datetime.date(2011, 3, 1),
        end_date=datetime.date(2012, 3, 31),
        initial_cutoff=212,
        num_validation=100,
        num_pp_iter=100,
        grid=250,
        n_sample_per_grid=20,
        domain=None,
        pp_class=pp_models.SeppStochasticNn,
        model_kwargs=None):

    if not model_kwargs:
        model_kwargs = {
            'max_delta_t': 150,
            'max_delta_d': 500,
            'bg_kde_kwargs': {'number_nn': [101, 16]},
            'trigger_kde_kwargs': {'number_nn': 15},
            'remove_conincident_pairs': False,
            'estimation_function': lambda x, y: estimation.estimator_exp_gaussian(x, y, 0.1, 50, frac_bg=None),
            'seed': 42,
        }

    if start_date + datetime.timedelta(days=initial_cutoff + num_validation) > end_date:
        warnings.warn("Requested number of validation runs is too large for the data size")

    if domain is None:
        domain = compute_chicago_region()

    data, t0, cid = get_crimes_by_type(
        crime_type='burglary',
        start_date=start_date,
        end_date=end_date,
        domain=domain,
    )

    sepp = pp_class(data=data, **model_kwargs)

    vb = validate.SeppValidationFixedModelIntegration(data, spatial_domain=domain, cutoff_t=initial_cutoff)
    vb.set_sample_units(grid, n_sample_per_grid=n_sample_per_grid)

    ## TODO: check the number of iterations reported is as expected here
    res = vb.run(time_step=1, n_iter=num_validation,
                 train_kwargs={'niter': num_pp_iter},
                 verbose=True)

    return res, vb


# TODO: need to update the call interface for this code
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


def validate_historic_kernel(start_date=datetime.datetime(2001, 3, 1, 0),
                           grid_size=250,
                           previous_n_days=7,
                           n_iter=20):

    pre_start_date = start_date - datetime.timedelta(days=previous_n_days)
    end_date = start_date + datetime.timedelta(days=n_iter)

    poly = compute_chicago_region()
    res, t0, cid = get_crimes_by_type(
        crime_type='burglary',
        start_date=pre_start_date,
        end_date=end_date
    )

    # use basic historic data spatial hotspot
    sk = hotspot.SKernelHistoric(previous_n_days)
    vb = validation.ValidationBase(res, sk, poly)
    vb.set_sample_units(grid_length=grid_size)
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
    res, t0, cid = get_crimes_by_type(
        crime_type='burglary',
        datetime__gte=pre_start_date
    )
    nslices = int(max(res[:, 0]) / dt)
    sk = hotspot.SKernelHistoric(previous_n_days)
    vb = validation.ValidationBase(res, sk, poly)
    vb.set_sample_units(grid_length=grid_size)
    vb.set_t_cutoff(previous_n_days)

    res = []

    for n in range(nslices):
        res.append(vb.run(dt=1, t_upper=previous_n_days + n_iter + n * dt))
        vb.set_t_cutoff(previous_n_days + (n + 1) * dt)

    return res


# implementation script
# test the effect of delta_t and delta_d on the output
# test_domain allows the restriction of testing to a subdomain (model is trained on full dataset either way)
def implement_delta_effect(outfile, test_domain=None):

    dts = [10, 20, 30, 40, 50, 60, 90]
    dds = [20, 50, 100, 200, 300, 500, 1000, 5000]

    start_date = datetime.datetime(2010, 1, 1, 0)
    end_date = datetime.datetime(2010, 7, 1, 0)
    niter = 30

    dt_grid, dd_grid = np.meshgrid(dts, dds)

    with open(outfile, 'w') as f:
        for dt, dd in zip(dt_grid.flat, dd_grid.flat):
            try:
                # train model
                tic = time()
                r, p = apply_point_process(
                    start_date=start_date,
                    end_date=end_date,
                    niter=niter,
                    max_delta_t=dt,
                    max_delta_d=dd,
                    pp_class=pp_models.SeppStochasticNn,
                )
                computation_time = time() - tic

                # validate model for next one month of data
                data, t0, cid = get_crimes_by_type(
                    start_date=end_date + datetime.timedelta(days=1),
                    end_date=end_date + datetime.timedelta(days=31),
                    domain=None
                )
                # disable parallel execution as it seems to slow things down here
                r.set_parallel(False)
                vb = validate.SeppValidationPreTrainedModel(data, copy.deepcopy(r), spatial_domain=test_domain)
                vb.set_t_cutoff(0)
                vb.set_sample_units(150)
                vres = vb.run(1)

                # only keep the first saved model for memory saving purposes
                del vres['model']

                this_res = {
                    'max_delta_t': dt,
                    'max_delta_d': dd,
                    'validation': vres,
                    'model': r,
                }

                dill.dump(this_res, f)

            except Exception:
                pass


# implementation script
# test the effect of spatial and temporal domain on the triggering function
def implement_spatial_temporal_domain_effect():

    niter = 35

    start_date_1 = datetime.datetime(2010, 1, 1, 0)
    start_date_2 = datetime.datetime(2010, 7, 1, 0)
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
    res, t0, cid = get_crimes_by_type(
        crime_type='burglary',
        start_date=start_date,
        end_date=end_date
    )

    vb = validate.PpValidation(res, spatial_domain=poly, model_kwargs={
        'max_trigger_t': 30,
        'max_trigger_d': 200,
        'estimator': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02),
    })
    vb.set_sample_units(250)
    vb.set_t_cutoff(first_training_size, b_train=False)

    sepp_res = vb.run(time_step=1, t_upper=first_training_size + 1,
                 train_kwargs={'niter': 20, 'tol_p': 1e-5},
                 verbose=True)

    # use basic historic data spatial hotspot
    sk = hotspot.SKernelHistoric(first_training_size) # use heatmap from same period
    vb_sk = validation.ValidationBase(res, sk, poly)
    vb_sk.roc.copy_sample_units(vb.roc)
    # vb_sk._grid = vb._grid
    # vb_sk.centroids = vb.centroids
    # vb_sk.a = vb.a
    vb_sk.set_t_cutoff(first_training_size, b_train=False)
    sk_res = vb_sk.run(time_step=1, t_upper=first_training_size + 1,
                         verbose=True)

    # use variable bandwidth KDE
    skvb = hotspot.SKernelHistoricVariableBandwidthNn(first_training_size)
    vb_skvb = validation.ValidationBase(res, skvb, poly)
    vb_skvb.roc.copy_sample_units(vb.roc)
    # vb_skvb._grid = vb._grid
    # vb_skvb.centroids = vb.centroids
    # vb_skvb.a = vb.a
    vb_skvb.set_t_cutoff(first_training_size, b_train=False)
    skvb_res = vb_skvb.run(time_step=1, t_upper=first_training_size + 1,
                         verbose=True)
