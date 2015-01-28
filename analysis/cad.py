__author__ = 'gabriel'
import warnings
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point as ShapelyPoint
from scipy.spatial.distance import pdist, squareform
import collections
from django.db.models import Q, Count, Sum, Min, Max
from django.contrib.gis.measure import D
from django.contrib.gis.geos import Polygon, MultiPolygon, LinearRing, Point
from plotting import geodjango_to_shapely, plot_geodjango_shapes
from database.views import month_iterator, week_iterator
import pandas
import numpy as np
import datetime
import pytz
from database import logic, models
from point_process import estimation, models as pp_models, validate
import hotspot
import validation
from itertools import combinations
from stats.logic import rook_boolean_connectivity, global_morans_i_p, local_morans_i as lmi
from django.db import connection
from database import osm

UK_TZ = pytz.timezone('Europe/London')
SEC_IN_DAY = float(24 * 60 * 60)

mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['interactive'] = True


def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=dt.tzinfo)
    delta = dt - epoch
    return delta.total_seconds()


def get_crimes_by_type(nicl_type=3, only_new=False, jiggle_scale=None, start_date=None,
                       end_date=None, spatial_domain=None):
    # Get CAD crimes by NICL type
    # data are de-duped, then processed into a (t, x, y) numpy array
    # times are in units of days, relative to t0

    if hasattr(nicl_type, '__iter__'):
        qset = []
        [qset.extend(logic.clean_dedupe_cad(nicl_type=t, only_new=only_new, spatial_domain=spatial_domain)) for t in nicl_type]
    else:
        qset = logic.clean_dedupe_cad(nicl_type=nicl_type, only_new=only_new, spatial_domain=spatial_domain)

    if start_date:
        start_date = start_date.replace(tzinfo=pytz.utc)
        qset = [x for x in qset if x.inc_datetime >= start_date]
    if end_date:
        end_date = end_date.replace(tzinfo=pytz.utc)
        qset = [x for x in qset if x.inc_datetime <= end_date]

    xy = np.array([x.att_map.coords for x in qset])
    t0 = np.min([x.inc_datetime for x in qset])
    t = np.array([[(x.inc_datetime - t0).total_seconds() / SEC_IN_DAY] for x in qset])
    if jiggle_scale:
        # xy = jiggle_all_points_on_grid(xy[:, 0], xy[:, 1])
        xy = jiggle_on_and_off_grid_points(xy[:, 0], xy[:, 1], scale=jiggle_scale)

    res = np.hstack((t, xy))

    # sort data
    sort_idx = np.argsort(res[:, 0])
    res = res[sort_idx]

    # indices
    cid = np.array([x.id for x in qset])[sort_idx]

    return res, t0, cid


def get_crimes_from_dump(table_name, spatial_domain=None, srid=27700):
    cur = connection.cursor()
    where_qry = ""
    if spatial_domain:
        where_qry += "WHERE ST_Within(location, ST_SetSRID(ST_GeomFromText('{0}'), {1}))".format(spatial_domain.wkt,
                                                                                                 srid)
    qry = """SELECT inc_date, ST_X(location), ST_Y(location), id FROM {0} {1}
             ORDER BY inc_date;""".format(
        table_name,
        where_qry
    )
    cur.execute(qry)
    res = cur.cursor.fetchall()
    t0 = res[0][0]
    t = [(r[0] - t0).total_seconds() / (24 * 60 * 60) for r in res]
    x = [r[1] for r in res]
    y = [r[2] for r in res]
    i = [r[3] for r in res]
    return np.vstack((t, x, y)).transpose(), t0, i


def dump_crimes_to_table(table_name,
                         nicl_type=3,
                         only_new=False,
                         jiggle_scale=None,
                         start_date=None,
                         end_date=None):
    """
    Extract crimes from the Django table and dump to a standalone table, with the same schema as that used for MA's data.
    NB the ID column has no meaning here, it's just autoincremented, so this cannot be compared with MA's data.
    """
    def parse_date(t, t0):
        dt = t0 + datetime.timedelta(days=t)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    cur = connection.cursor()
    res, t0, cid = get_crimes_by_type(nicl_type=nicl_type,
                                 only_new=only_new,
                                 jiggle_scale=jiggle_scale,
                                 start_date=start_date,
                                 end_date=end_date)
    drop_sql = """DROP TABLE {0};""".format(table_name)
    create_sql = """CREATE TABLE {0} (id SERIAL PRIMARY KEY, inc_datetime TIMESTAMP);
                    SELECT AddGeometryColumn('{0}', 'location', 27700, 'POINT', 2);""".format(table_name)

    try:
        cur.execute(drop_sql)
    except Exception as exc:
        print repr(exc)
    cur.execute(create_sql)

    pt_from_text_sql = """ST_GeomFromText('POINT(%f %f)', 27700)"""
    for i in range(len(res)):
        x = res[i]
        insert_sql = """INSERT INTO {0} (id, inc_datetime, location) VALUES ({1}, '{2}', {3});""".format(
            table_name,
            cid[i],
            parse_date(x[0], t0),
            pt_from_text_sql % (x[1], x[2])
        )
        cur.execute(insert_sql)


def get_camden_region():
    camden = models.Division.objects.get(type='borough', name__iexact='camden')
    return camden.mpoly.simplify()  # type Polygon


class CadAggregate(object):
    def __init__(self, nicl_number=None, only_new=False, start_date=None, end_date=None):
        self._start_date = start_date
        self._end_date = end_date
        self.nicl_number = nicl_number
        if nicl_number:
            if hasattr(nicl_number, '__iter__'):
                self.nicl_name = [t.description for t in models.Nicl.objects.filter(number__in=nicl_number)]
            else:
                self.nicl_name = models.Nicl.objects.get(number=nicl_number).description
        else:
            self.nicl_name = 'All crime types'
        self.only_new = only_new
        self.cad = None
        self.load_data()

    def load_data(self):

        self.cad, t0, cid = get_crimes_by_type(nicl_type=self.nicl_number, only_new=self.only_new,
                                               start_date=self._start_date, end_date=self._end_date)


        # self.cad = logic.initial_filter_cad(nicl_type=self.nicl_number, only_new=self.only_new)
        # if self.start_date:
        #     self.cad = self.cad.filter(inc_datetime__gte=self.start_date)
        # if self.end_date:
        #     self.cad = self.cad.filter(inc_datetime__lte=self.end_date)
        # if self.dedupe:
        #     self.cad = logic.dedupe_cad(self.cad)

    @property
    def start_date(self):
        if self._start_date:
            return self._start_date
        return min([x.inc_datetime for x in self.cad]).replace(hour=0, minute=0, second=0)

    @property
    def end_date(self):
        if self._end_date:
            return self._end_date
        return max([x.inc_datetime for x in self.cad])

    def aggregate(self):
        raise NotImplementedError()


class CadSpatialGrid(CadAggregate):
    def __init__(self, grid=None, **kwargs):
        # defer dedupe until after spatial aggregation
        super(CadSpatialGrid, self).__init__(**kwargs)
        # load grid or use one provided
        self.grid = grid or models.Division.objects.filter(type='cad_250m_grid')
        self.shapely_grid = [geodjango_to_shapely(x.mpoly)[0] for x in self.grid]
        # self.shapely_grid = pandas.Series([geodjango_to_shapely([x.mpoly]) for x in self.grid],
        #                                   index=[x.name for x in self.grid])
        # gridded data
        self.data = self.aggregate()

    def aggregate(self):
        working_cad = np.array(self.cad)
        res = collections.OrderedDict()
        for g in self.shapely_grid:
            # res[g] = np.array([(t, x, y) for (t, x, y) in self.cad if ShapelyPoint(x, y).within(g)])
            in_grid = []
            idx = []
            if working_cad.ndim == 1:
                import ipdb; ipdb.set_trace()
            for i in range(working_cad.shape[0]):
                pt = ShapelyPoint(working_cad[i, 1], working_cad[i, 2])
                if pt.intersects(g):
                    in_grid.append(working_cad[i, :])
                    idx.append(i)

            res[g] = np.array(in_grid)
            working_cad = np.delete(working_cad, idx, axis=0)


        return res
        # return logic.cad_aggregate_grid(self.cad, grid=self.grid)


class CadTemporalAggregation(CadAggregate):
    def __init__(self, **kwargs):
        super(CadTemporalAggregation, self).__init__(**kwargs)
        bucket_dict = self.bucket_dict()
        self.data = []
        self.data = self.bucket_data()

    def bucket_dict(self):
        return {'all': lambda x: True}

    def bucket_data(self):
        return logic.time_aggregate_data(self.cad, bucket_dict=self.bucket_dict())


class CadDaily(CadTemporalAggregation):
    @staticmethod
    def create_bucket_fun(sd, ed):
        """ Interesting! If we just generate the lambda function inline in bucket_dict, the scope is updated each time,
         so the parameters (sd, ed) are also updated.  In essence the final function created is run every time.
         Instead use a function factory to capture the closure correctly. """
        return lambda x: sd <= x.inc_datetime < ed

    def bucket_dict(self):
        gen_day = logic.n_day_iterator(self.start_date, self.end_date)
        return collections.OrderedDict(
            [(sd, self.create_bucket_fun(sd, ed)) for sd, ed in gen_day]
        )



class CadNightDay(CadTemporalAggregation):
    @staticmethod
    def create_bucket_fun(st, et):
        """ Interesting! If we just generate the lambda function inline in bucket_dict, the scope is updated each time,
         so the parameters (sd, ed) are also updated.  In essence the final function created is run every time.
         Instead use a function factory to capture the closure correctly. """
        return lambda x: st <= x.inc_datetime.time() < et

    def bucket_dict(self):
        return collections.OrderedDict([
            ('daytime', lambda x: datetime.time(6, 0, 0) <= x.inc_datetime.time() < datetime.time(18, 0, 0)),
            ('nighttime', lambda x: datetime.time(6, 0, 0) > x.inc_datetime.time()
                                    or x.inc_datetime.time() >= datetime.time(18, 0, 0)),
        ]
        )

def night_day_split_for_chris_gale():
    import csv
    res = []
    for i in range(15):  # all major crime types
        res.append(CadNightDay(nicl_number=i+1))

    # create CSV
    header = ['NICL number', 'NICL name', 'day counts', 'night counts', 'total counts']
    with open('for_chris.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(header)
        for row in res:
            nt = len(row.data['nighttime'])
            dt = len(row.data['daytime'])
            datum = [row.nicl_number, row.nicl_name, nt, dt, nt + dt]
            c.writerow(datum)


def cad_space_time_count(cad_temporal, cad_spatial):
    # create numpy array with counts
    res = logic.combine_aggregations_into_count(cad_temporal.data, cad_spatial.data)
    return pandas.DataFrame(res, index=cad_temporal.data.keys(), columns=cad_spatial.data.keys())


def cad_spatial_repeat_analysis(nicl_type=3):
    qset = logic.clean_dedupe_cad(nicl_type=nicl_type)
    divs = models.Division.objects.filter(type='cad_250m_grid')
    centroids = np.array([t.centroid.coords for t in divs.centroid()])
    xy = np.array([t.att_map.coords for t in qset])
    on_grid = collections.defaultdict(int)
    off_grid_rpt = collections.defaultdict(int)
    off_grid = collections.defaultdict(int)
    for t in xy:
        if np.any(np.sum(centroids == t, axis=1) == 2):
            on_grid[tuple(t)] += 1
            continue
        if np.sum((np.sum(xy == t, axis=1) == 2)) > 1:
            off_grid_rpt[tuple(t)] += 1
        else:
            off_grid[tuple(t)] += 1

    # plotting
    camden = models.Division.objects.get(name='Camden', type='borough')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # camden outline
    h = plot_geodjango_shapes([camden.mpoly], ax=ax)
    [x[0].set_facecolor('none') for x in h]

    # grid divisions
    h = plot_geodjango_shapes([x.mpoly for x in divs], ax=ax)
    [x[0].set_facecolor('none') for x in h]
    [x[0].set_edgecolor('#CCCCCC') for x in h[1:]]

    # grid centroid repeat locations
    ong = np.array(on_grid.keys())
    ax.plot(ong[:, 0], ong[:, 1], 'or')

    # off-grid repeat locations
    offg = np.array(off_grid_rpt.keys())
    ax.plot(offg[:, 0], offg[:, 1], 'ok')

    # off-grid unique locations
    offgu = np.array(off_grid.keys())
    ax.plot(offgu[:, 0], offgu[:, 1], 'o', color='#CCCCCC', alpha=0.6)

    x_max, y_max = np.max(np.array(camden.mpoly[0].coords[0]), axis=0)
    x_min, y_min = np.min(np.array(camden.mpoly[0].coords[0]), axis=0)

    ax.set_xlim(np.array([-150, 150]) + np.array([x_min, x_max]))
    ax.set_ylim(np.array([-150, 150]) + np.array([y_min, y_max]))
    ax.set_aspect('equal')

    plt.draw()

    return on_grid, off_grid_rpt, off_grid


def jiggle_on_grid_points(x, y):
    """ randomly distribute points located at the centroid of the grid squares in order to avoid the issues arising from
        spurious exact repeats. """
    divs = models.Division.objects.filter(type='cad_250m_grid')
    centroids = np.array([t.centroid.coords for t in divs.centroid()])
    res = []
    for t in zip(x, y):
        nt = t
        if np.any(np.sum(centroids == t, axis=1) == 2):
            # pick new coords
            nt = (np.random.random(2) * 250 - 125) + t
            # print t, " -> ", nt
        res.append(nt)
    return np.array(res)


def jiggle_on_and_off_grid_points(x, y, scale=5):
    """ randomly jiggle points that are
        a) off-grid but non-unique, in order to avoid exact overlaps
        b) on-grid, in which case move them randomly around inside the grid square
        scale parameter is the symmetric bivariate normal bandwidth for the off-grid jiggle. """
    divs = models.Division.objects.filter(type='cad_250m_grid')
    centroids = np.array([t.centroid.coords for t in divs.centroid()])
    xy = np.vstack((x, y)).transpose()
    res = []
    for t in zip(x, y):
        nt = t
        if np.any(np.sum(centroids == t, axis=1) == 2):
            nt = (np.random.random(2) * 250 - 125) + t
        elif np.sum((np.sum(xy == t, axis=1) == 2)) > 1:
            nt = np.random.normal(loc=0., scale=scale, size=(2,)) + t
        else:
            nt = np.array(t)
        res.append(nt)
    return np.array(res)


def jiggle_all_points_on_grid(x, y):
    """ randomly distribute points located at the centroid of the grid squares in order to avoid the issues arising from
        spurious exact repeats. """
    divs = models.Division.objects.filter(type='cad_250m_grid')
    extents = np.array([t.mpoly.extent for t in divs])
    ingrid = lambda t: np.where(
        (t[0] >= extents[:, 0]) &
        (t[0] < extents[:, 2]) &
        (t[1] >= extents[:, 1]) &
        (t[1] < extents[:, 3])
    )[0][0]
    res = []
    for t in zip(x, y):
        # find grid square
        try:
            idx = ingrid(t)
        except IndexError:
            # not on grid - leave as-is
            warnings.warn("Point found that is not on the CAD grid.  Leaving as-is.")
            nt = t
        else:
            e = extents[idx]
            # jiggle
            nt = np.random.random(2) * 250 + e[:2]
        res.append(nt)
    return np.array(res)


def apply_sepp_to_data(data,
                       max_delta_t,
                       max_delta_d,
                       estimation_function,
                       niter=50,
                       bg_kde_kwargs=None,
                       trigger_kde_kwargs=None,
                       sepp_class=pp_models.SeppStochasticNnReflected,
                       rng_seed=42,
                       ):
    bg_kde_kwargs = bg_kde_kwargs or {}
    trigger_kde_kwargs = trigger_kde_kwargs or {}
    r = sepp_class(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                   bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs,
                   estimation_function=estimation_function)
    if rng_seed:
        r.set_seed(rng_seed)
    r.train(niter=niter)
    return r


def apply_point_process(nicl_type=3,
                        only_new=False,
                        start_date=None,
                        end_date=None,
                        niter=15,
                        num_nn=None,
                        min_bandwidth=None,
                        jiggle_scale=None,
                        max_delta_t=60,  # days
                        max_delta_d=500,  # metres
                        sepp_class=pp_models.SeppStochasticNnReflected,
                        tol_p=None,
                        data=None,
                        rng_seed=42,
                        ):

    # suggested value:
    # min_bandwidth = np.array([0.3, 5., 5.])

    # get data
    if data is not None:
        res = data
    else:
        res, t0, cid = get_crimes_by_type(nicl_type=nicl_type, only_new=only_new, jiggle_scale=jiggle_scale,
                                          start_date=start_date, end_date=end_date)

    # define initial estimator
    est = lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02)

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
        'strict': False  # attempt to restore order even if number of BG becomes less than requested NNs
    }

    trigger_kde_kwargs = {
        'min_bandwidth': min_bandwidth,
        'number_nn': num_nn_trig,
        'strict': False  # attempt to restore order even if number of trig becomes less than requested NNs
    }

    return apply_sepp_to_data(
        res,
        max_delta_t=max_delta_t,
        max_delta_d=max_delta_d,
        estimation_function=est,
        niter=niter,
        bg_kde_kwargs=bg_kde_kwargs,
        trigger_kde_kwargs=trigger_kde_kwargs,
        sepp_class=sepp_class,
        rng_seed=rng_seed
    )

    # r = sepp_class(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = sepp_class(data=res, max_delta_d=max_delta_d, max_delta_t=max_delta_t)
    # r.p = estimation.estimator_bowers(res, r.linkage, ct=1, cd=0.02)
    #
    # # train on all data
    # ps = r.train(niter=niter, tol_p=tol_p)
    # return r, ps


def apply_sepp_to_tabular_data(table_name):
    data, t0 = get_crimes_from_dump(table_name)
    est = lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02)
    max_delta_t = 60
    max_delta_d = 500
    niter = 50
    bg_kde_kwargs = {
        'number_nn': [100, 15],
    }

    trigger_kde_kwargs = {
        'min_bandwidth': [0.5, 10, 10],
        'number_nn': 15,
    }
    sepp_class = pp_models.SeppStochasticNnReflected

    # filter data to provide correct quantity for training
    # data = data[data[:, 0] >= 151.]
    data = data[data[:, 0] <= 210.]

    r = apply_sepp_to_data(
        data,
        max_delta_t=max_delta_t,
        max_delta_d=max_delta_d,
        estimation_function=est,
        niter=niter,
        bg_kde_kwargs=bg_kde_kwargs,
        trigger_kde_kwargs=trigger_kde_kwargs,
        sepp_class=sepp_class
    )

    return r


def validate_point_process(
        nicl_type=3,
        end_date=datetime.datetime(2012, 3, 1, tzinfo=pytz.utc),
        start_date=None,
        jiggle=None,
        num_validation=10,
        num_pp_iter=15,
        grid=100,
        time_step=1,
        pred_dt_plus=1,
        ):

    # get data
    res, t0, cid = get_crimes_by_type(nicl_type=nicl_type, only_new=True, jiggle_scale=jiggle, start_date=start_date)

    # find end_date in days from t0
    end_days = (end_date - t0).total_seconds() / SEC_IN_DAY

    # get domain
    poly = get_camden_region()

    vb = validate.SeppValidation(res, spatial_domain=poly, model_kwargs={
        'max_delta_t': 60,
        'max_delta_d': 1000,
        'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=0.02),
        'trigger_kde_kwargs': {'min_bandwidth': np.array([0.3, 5., 5.])},
    })
    vb.set_grid(grid)
    vb.set_t_cutoff(end_days, b_train=False)

    res = vb.run(time_step=time_step, t_upper=end_days + num_validation, pred_dt_plus=pred_dt_plus,
                 train_kwargs={'niter': num_pp_iter},
                 verbose=True)

    return res, vb


def validate_historic_kde(
        nicl_type=3,
        end_date=datetime.datetime(2012, 3, 1, tzinfo=pytz.utc),
        start_date=None,
        num_validation=30,
        kind=None,
        grid=100,
        prediction_dt=1,
        pred_dt_plus=1):

    # kind keyword specifies the kind of kernel to use
    if not kind or kind == 'fbk':
        sk = hotspot.SKernelHistoric()
    elif kind == 'nnk':
        sk = hotspot.SKernelHistoricVariableBandwidthNn()
    else:
        raise AttributeError("Specified 'kind' argument not recognised")

    # get data
    res, t0, cid = get_crimes_by_type(nicl_type=nicl_type, only_new=True, jiggle_scale=None, start_date=start_date)

    # find end_date in days from t0
    end_days = (end_date - t0).total_seconds() / SEC_IN_DAY

    # get domain
    poly = get_camden_region()



    vb = validation.ValidationBase(res, hotspot.Hotspot, poly, model_args=(sk,))
    vb.set_grid(grid)
    vb.set_t_cutoff(end_days, b_train=False)

    res = vb.run(time_step=prediction_dt, t_upper=end_days + num_validation, pred_dt_plus=pred_dt_plus,
                 verbose=True)

    return res, vb


def diggle_st_clustering(nicl_type=3):
    ## TODO: TEST ME
    qset = logic.clean_dedupe_cad(nicl_type=nicl_type)
    rel_dt = np.min([x.inc_datetime for x in qset])
    res = np.array([[(x.inc_datetime - rel_dt).total_seconds()] + list(x.att_map.coords) for x in qset])
    # get combined domain of all grid squares
    domain = models.Division.objects.filter(type='cad_250m_grid').unionagg()
    A = domain.area
    T = np.max(res[:, 0])
    n = res.shape[0]
    t2, t1 = np.meshgrid(res[:, 0], res[:, 0], copy=True)
    x2, x1 = np.meshgrid(res[:, 1], res[:, 1], copy=True)
    y2, y1 = np.meshgrid(res[:, 2], res[:, 2], copy=True)
    u = np.abs(t2 - t1)
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    v_tmp = np.repeat(res[:, 0], n).reshape((n, n))
    v = np.ones((n, n))
    v[(v_tmp - u) < 0] = 2
    v[(v_tmp + u) > T] = 2
    v[np.diag_indices(n)] = 0

    ## TODO: this is SO SLOW
    w = np.ones((n, n))
    for i in range(n):
        print i
        centre = res[i, 1:3]
        for j in range(n):
            if i== j:
                w[i, j] = 0
                continue
            radius = d[i, j]
            if d[i, j] < 1e-12:
                # exact repeat or same grid snapping
                w[i, j] = 1.
                continue
            circ = Point(*centre).buffer(radius).exterior_ring
            l = domain.intersection(circ).length
            w[i, j] = 2 * np.pi * radius / l


def daily_iterator(start_date, end_date):
    d = start_date
    while True:
        if d > end_date:
            raise StopIteration
        yield (d, d + datetime.timedelta(days=1))
        d += datetime.timedelta(days=1)


class CadByGrid(object):

    def __init__(self, data, t0, data_index=None, grid=None):
        """

        :param data: N x 3 np.ndarray or DataArray (t, x, y), all float
        :param data_index: length N array containing indices corresponding to data.  If not supplied, lookup index used.
        :param t0: datetime corresponding to first record
        :param grid: optional list of grid (multi)polygons, default is CAD 250m grid
        :return:
        """

        self.grid = grid or models.Division.objects.filter(type='cad_250m_grid')
        self.shapely_grid = [geodjango_to_shapely(x)[0] for x in self.grid]
        # preliminary cad filter
        self.data = data
        self.t0 = t0
        self.data_index = np.array(data_index) if data_index is not None else np.arange(len(data))

        self.dates, self.indices = self.compute_array()

    @property
    def ngrid(self):
        return len(self.grid)

    @property
    def start_date(self):
        sd = self.data[:, 0].min()
        return self.t0 + datetime.timedelta(days=sd)

    @property
    def end_date(self):
        ed = self.data[:, 0].max()
        return self.t0 + datetime.timedelta(days=ed)

    def compute_array(self):
        dates = [[] for y in range(self.ngrid)]
        indices = [[] for y in range(self.ngrid)]


        # iterate over grid
        for j in range(self.ngrid):
            this_grid = self.grid[j]
            this_extent = this_grid.extent

            this_idx = (this_extent[0] <= self.data[:, 1]) & (self.data[:, 1] < this_extent[2]) &\
                       (this_extent[1] <= self.data[:, 2]) & (self.data[:, 2] < this_extent[3])

            these_dates = self.data[this_idx, 0]
            these_inds = self.data_index[this_idx]

            # convert dates from float to datetimes
            these_dates = [self.t0 + datetime.timedelta(days=x) for x in these_dates]

            dates[j] = these_dates
            indices[j] = these_inds

        return dates, indices

    def all_time_aggregate(self):
        bucket_fun = lambda x: True
        return self.time_aggregate_data({'all': bucket_fun})


    def weekday_weekend_aggregate(self):
        bucket_dict = collections.OrderedDict(
            [
                ('Weekday', lambda x: x.weekday() < 5),
                ('Weekend', lambda x: x.weekday() >= 5),
            ]
        )
        return self.time_aggregate_data(bucket_dict)

    def daytime_evening_aggregate(self):
        am = datetime.time(6, 0, 0, tzinfo=UK_TZ)
        pm = datetime.time(18, 0, 0, tzinfo=UK_TZ)
        bucket_dict = collections.OrderedDict(
            [
                ('Daytime', lambda x: am <= x.time() < pm),
                ('Evening', lambda x: (pm <= x.time()) or (x.time() < am)),
            ]
        )
        return self.time_aggregate_data(bucket_dict)

    @staticmethod
    def create_bucket_fun(sd, ed):
        """ Interesting! If we just generate the lambda function inline in bucket_dict, the scope is updated each time,
         so the parameters (sd, ed) are also updated.  In essence the final function created is run every time.
         Instead use a function factory to capture the closure correctly. """
        return lambda x: sd <= x < ed

    def daily_aggregate_data(self):
        sd = self.start_date
        ed = self.end_date
        g = daily_iterator(sd, ed)
        bucket_dict = collections.OrderedDict()
        for a, b in g:
            bucket_dict[a.strftime('%Y-%m-%d')] = self.create_bucket_fun(a, b)
        return bucket_dict

    def time_aggregate_data(self, bucket_dict):

        data = np.zeros((len(bucket_dict), self.ngrid))
        for j in range(self.ngrid): # grid squares
            for k, func in enumerate(bucket_dict.values()): # time buckets
                data[k, j] = len([x for x in self.dates[j] if func(x)])

        return pandas.DataFrame(data, index=bucket_dict.keys())


def global_i_analysis():

    short_names = ['Violence', 'Sexual Offences', 'Burglary Dwelling', 'Burglary Non-dwelling',
                   'Robbery', 'Theft of Vehicle', 'Theft from Vehicle', 'Other Theft',
                   'Fraud and Forgery', 'Criminal Damage', 'Drug Offences', 'Bomb Threat',
                   'Shoplifting', 'Harassment', 'Abduction/Kidnap']

    cbg = CadByGrid()
    a = cbg.all_time_aggregate()

    # sort by ascending number of crimes
    sort_idx = np.argsort(a.sum().values)
    short_names = [short_names[i] for i in sort_idx]
    a = a[sort_idx]

    W = rook_boolean_connectivity(cbg.grid)
    global_i = [(x, global_morans_i_p(a[x], W, n_iter=5000)) for x in a]

    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.75])
    hbar = ax.bar(range(cbg.l), [x[1][0] for x in global_i], width=0.8, edgecolor='k')
    for i in range(cbg.l):
        if global_i[i][1][1] < 0.01:
            hbar[i].set_facecolor('#FF9C9C')
        elif global_i[i][1][1] < 0.05:
            hbar[i].set_facecolor('#FFFF66')
        else:
            hbar[i].set_facecolor('gray')
    ax.set_xticks([float(x) + 0.5 for x in range(cbg.l)])
    ax.set_xticklabels(short_names, rotation=60, ha='right', fontsize=18)
    ax.set_ylabel('Global Moran''s I', fontsize=22)
    ax.set_xlim([0, cbg.l])

    yticks = ax.yaxis.get_ticklabels()
    plt.setp(yticks, fontsize=20)
    plt.show()


def numbers_by_type():

    short_names = ['Violence', 'Sexual Offences', 'Burglary Dwelling', 'Burglary Non-dwelling',
                   'Robbery', 'Theft of Vehicle', 'Theft from Vehicle', 'Other Theft',
                   'Fraud and Forgery', 'Criminal Damage', 'Drug Offences', 'Bomb Threat',
                   'Shoplifting', 'Harassment', 'Abduction/Kidnap']

    cbg = CadByGrid()
    bucket_dict = collections.OrderedDict(
        [
            ('all', lambda x: True),
            ('old', lambda x: x < logic.CAD_GEO_CUTOFF),
            ('new', lambda x: x >= logic.CAD_GEO_CUTOFF),
        ]
    )
    res = cbg.time_aggregate_data(bucket_dict)
    num_crimes_all = res['all'].sum()
    sort_idx = np.argsort(num_crimes_all.values)
    num_crimes_old = res['old'].sum()[sort_idx]
    num_crimes_new = res['new'].sum()[sort_idx]
    short_names = [short_names[i] for i in sort_idx]

    # FIGURE: crime numbers split by old/new

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.75])
    hbar_old = ax.bar(range(cbg.l), num_crimes_old.values, width=0.8, color='black')
    hbar_new = ax.bar(range(cbg.l), num_crimes_new.values, width=0.8, color='gray', bottom=num_crimes_old.values)

    ax.set_xticks([float(x) + 0.5 for x in range(cbg.l)])
    ax.set_xticklabels(short_names)

    plt.legend(('Snapped to grid', 'Snapped to segment'), loc='upper left')

    xticks = ax.xaxis.get_ticklabels()
    plt.setp(xticks, rotation=60, ha='right', fontsize=18)
    yticks = ax.yaxis.get_ticklabels()
    plt.setp(yticks, fontsize=20)
    ax.set_xlim([0, cbg.l])

    # FIGURE: crime numbers unsplit
    num_crimes_both = num_crimes_old + num_crimes_new

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.75])
    ax.bar(range(cbg.l), num_crimes_both.values, width=0.8, color='black')

    ax.set_xticks([float(x) + 0.5 for x in range(cbg.l)])
    ax.set_xticklabels(short_names)

    xticks = ax.xaxis.get_ticklabels()
    plt.setp(xticks, rotation=60, ha='right', fontsize=18)
    yticks = ax.yaxis.get_ticklabels()
    plt.setp(yticks, fontsize=20)
    ax.set_xlim([0, cbg.l])

    plt.show()


def spatial_density_all_time_all_crimes():

    poly = get_camden_region()
    camden_mpoly = geodjango_to_shapely([get_camden_region()])
    oo = osm.OsmRendererBase(poly, buffer=100)

    cbg = CadByGrid()
    a = cbg.all_time_aggregate()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=ccrs.OSGB())
    ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
    ax.background_patch.set_visible(False)
    oo.render(ax)

    a = a.sum(axis=1)

    cmap = mpl.cm.Reds
    norm = mpl.colors.Normalize()
    norm.autoscale(a)
    cax = mpl.colorbar.make_axes(ax, location='bottom', pad=0.02, fraction=0.05, shrink=0.9)
    cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for j in range(cbg.m):
        val = a.values[j]
        fc = sm.to_rgba(val) if val else 'none'
        ax.add_geometries(geodjango_to_shapely([cbg.grid[j].mpoly]), ccrs.OSGB(), facecolor=fc, alpha=0.3)

    ax.add_geometries(camden_mpoly, ccrs.OSGB(), facecolor='none', edgecolor='black')

    plt.show()


def spatial_density_all_time_by_crime(nicl_numbers=None, short_names=None):

    nicl_numbers = nicl_numbers or [3, 6, 10]
    n = len(nicl_numbers)
    cbg = {}

    for i in range(n):
        this_cbg = CadSpatialGrid(nicl_number=nicl_numbers[i])
        cbg[nicl_numbers[i]] = this_cbg

    return cbg


def plot_spatial_density_all_time_by_crime(cad_spatial_grid):
    n = len(cad_spatial_grid)
    poly = get_camden_region()
    xmin, ymin, xmax, ymax = poly.buffer(100).extent
    oo = osm.OsmRendererBase(poly, buffer=100)
    fig = plt.figure(figsize=(15, 6))
    axes = [fig.add_subplot(1, n, i+1, projection=ccrs.OSGB()) for i in range(n)]
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.03, hspace=0.01)
    count = 0
    for i, this_cbg in enumerate(cad_spatial_grid.itervalues()):
        ax = axes[i]
        oo.render(ax)
        ds = [len(t) for t in this_cbg.data.values()]
        # ax.set_title(short_names[i])
        ax.set_extent([xmin, xmax, ymin, ymax], ccrs.OSGB())
        ax.background_patch.set_visible(False)
        # ax.outline_patch.set_visible(False)
        cmap = mpl.cm.autumn
        norm = mpl.colors.Normalize(0, max(ds))
        # norm.autoscale(ds)
        cax = mpl.colorbar.make_axes(ax, location='bottom', pad=0.02, fraction=0.05, shrink=0.9)
        cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for j in range(len(this_cbg.grid)):
            val = ds[j]
            fc = sm.to_rgba(val) if val else 'none'
            ax.add_geometries(this_cbg.shapely_grid[j:j+1], ccrs.OSGB(), facecolor=fc)

        # ax.add_geometries(geodjango_to_shapely(poly), ccrs.OSGB(), facecolor='none', edgecolor='black')

    plt.show()



def spatial_density_weekday_evening():

    nicl_numbers = [3, 6, 10]
    short_names = ['Burglary Dwelling', 'Veh Theft', 'Crim Damage']
    camden_mpoly = geodjango_to_shapely([models.Division.objects.get(name='Camden', type='borough').mpoly])
    cbg = CadByGrid(nicl_numbers=nicl_numbers)
    a = cbg.weekday_weekend_aggregate()
    b = cbg.daytime_evening_aggregate()
    # combine
    for x in b.items:
        a[x] = b[x]
    b = a.transpose(2, 0, 1)

    for i in range(cbg.l): # crime types

        df = b.iloc[i]
        n = df.shape[0]
        fig = plt.figure(figsize=(20, 6))
        axes = [fig.add_subplot(1, n, t+1, projection=ccrs.OSGB()) for t in range(n)]
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.03, hspace=0.01)

        for j in range(n):
            ax = axes[j]
            ds = df.iloc[j]
            ax.set_title(ds.name)
            ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
            ax.background_patch.set_visible(False)
            # ax.outline_patch.set_visible(False)

            cmap = mpl.cm.cool
            norm = mpl.colors.Normalize()
            norm.autoscale(ds)
            cax = mpl.colorbar.make_axes(ax, location='bottom', pad=0.02, fraction=0.05, shrink=0.9)
            cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

            for j in range(cbg.m):
                val = ds.values[j]
                fc = sm.to_rgba(val) if val else 'none'
                ax.add_geometries(geodjango_to_shapely([cbg.grid[j].mpoly]), ccrs.OSGB(), facecolor=fc)

            ax.add_geometries(camden_mpoly, ccrs.OSGB(), facecolor='none', edgecolor='black')

    plt.show()


def local_morans_i():

    nicl_numbers = [1, 3, (6, 7)]
    short_names = ['Violence', 'Burglary dwelling', 'Theft of/from vehicle']
    camden_mpoly = geodjango_to_shapely([models.Division.objects.get(name='Camden', type='borough').mpoly])
    cbg = CadByGrid(nicl_numbers=nicl_numbers)
    a = cbg.all_time_aggregate()
    W = rook_boolean_connectivity(cbg.grid)

    fig = plt.figure(figsize=(15, 6))
    axes = [fig.add_subplot(1, cbg.l, i+1, projection=ccrs.OSGB()) for i in range(cbg.l)]
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.03, hspace=0.01)

    local_i = []
    standard_local_i = []
    max_li = 0.
    min_li = 1e6

    for i in range(cbg.l):
        ds = a[cbg.nicl_names[i]]
        this_res = lmi(ds, W)
        local_i.append(this_res)
        standard_local_i.append(this_res / this_res.std())
        max_li = max(max_li, max(standard_local_i[i]))
        min_li = min(min_li, min(standard_local_i[i]))

    for i in range(cbg.l):
        ax = axes[i]
        ax.set_title(short_names[i])
        ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
        ax.background_patch.set_visible(False)
        # ax.outline_patch.set_visible(False)
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=min_li, vmax=max_li)
        # norm.autoscale(local_i[i])
        cax = mpl.colorbar.make_axes(ax, location='bottom', pad=0.02, fraction=0.05, shrink=0.9)
        cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for j in range(cbg.m):
            val = standard_local_i[i].values[j]
            fc = sm.to_rgba(val) if val else 'none'
            ax.add_geometries(geodjango_to_shapely([cbg.grid[j].mpoly]), ccrs.OSGB(), facecolor=fc)

        ax.add_geometries(camden_mpoly, ccrs.OSGB(), facecolor='none', edgecolor='black')

    plt.show()

    return local_i


def pairwise_distance(nicl_number=3):
    cad = logic.initial_filter_cad()
    cad = cad.filter(Q(cl01=nicl_number) | Q(cl02=nicl_number) | Q(cl03=nicl_number)).distinct('cris_entry')
    cad = sorted(cad, key=lambda x: x.inc_datetime)
    coords = np.array([x.att_map.coords for x in cad])
    ## FIXME: this is in units of metres, but that's only because we know the coordinate system is OSGB36?
    return pdist(coords)
    # p = squareform(pdist(coords))
    # index = [x.id for x in cad]
    # return cad, pandas.DataFrame(p, index=index, columns=index)


def pairwise_time_difference(nicl_number=3):

    cad = logic.initial_filter_cad()
    cad = cad.filter(Q(cl01=nicl_number) | Q(cl02=nicl_number) | Q(cl03=nicl_number)).distinct('cris_entry')
    cad = sorted(cad, key=lambda x: x.inc_datetime)

    times = [x.inc_datetime for x in cad]
    cond_vec = np.array([(x[1]-x[0]).days for x in combinations(times, 2)])
    return cond_vec
    # p = squareform(cond_vec)
    # index = [x.id for x in cad]
    # return list(cad), pandas.DataFrame(p, index=index, columns=index)


def condensed_to_index(sub, n):
    idx = np.triu_indices(n)
    try:
        return [(idx[0][i], idx[1][i]) for i in idx]
    except TypeError:
        return (idx[0][sub], idx[1][sub])


def pairwise_time_lag_events(max_distance=200, nicl_numbers=3, num_bins=None):
    """ Recreate Fig 1(b) Mohler et al 2011 'Self-exciting point process modeling of crime'
        max_distance is in units of metres. """

    cad, t0, cid = get_crimes_by_type(nicl_type=nicl_numbers)
    x1, x0 = np.meshgrid(cad[:, 1], cad[:, 1], copy=False)
    y1, y0 = np.meshgrid(cad[:, 2], cad[:, 2], copy=False)
    t1, t0 = np.meshgrid(cad[:, 0], cad[:, 0], copy=False)
    triu_idx = np.triu_indices_from(x0, k=1)
    dx = x1[triu_idx] - x0[triu_idx]
    dy = y1[triu_idx] - y0[triu_idx]
    pt = t1[triu_idx] - t0[triu_idx]
    pd = np.sqrt(dx **2 + dy ** 2)

    filt_sub = np.where(pd < max_distance)[0]
    time_diffs = pt[filt_sub]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_win = int(np.max(time_diffs))
    D = 0.5 * n_win * (n_win - 1)
    # import ipdb; ipdb.set_trace()
    ax.hist(time_diffs, num_bins or range(n_win), normed=True, edgecolor='none', facecolor='gray')
    ax.plot(np.arange(1, n_win), np.arange(1, n_win)[::-1]/D, 'k--', lw=2)
    ax.set_xlabel('Time difference (days)')
    ax.set_ylabel('Event pair density')

    return time_diffs


def pairwise_distance_events(max_time=14, nicl_numbers=None, num_bins=50):
    """ Recreate Fig 2 Mohler et al 2011 'Self-exciting point process modeling of crime'
        except that we look at the distribution of spatial distance and fix time window. """

    nicl_numbers = nicl_numbers or [1, 3, 13]
    n = len(nicl_numbers)
    res = []
    cad = logic.initial_filter_cad(only_new=True)
    for nicl in nicl_numbers:
        pd = pairwise_distance(nicl)
        pt = pairwise_time_difference(nicl)
        filt_sub = np.where(pt < max_time)[0]
        space_diffs = pd[filt_sub]
        res.append(space_diffs)

    fig = plt.figure()
    for i in range(n):
        ax = fig.add_subplot(1, n, i)
        # n_win = int(max(res[i]))
        # D = 0.5 * n_win * (n_win - 1)
        ax.hist(res[i], num_bins, normed=True, edgecolor='none', facecolor='gray')
        # ax.plot(np.arange(1, n_win), np.arange(1, n_win)[::-1]/D, 'k--')
        ax.set_xlabel('Separation distance')
        ax.set_ylabel('Event pair density')

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, wspace=0.03, hspace=0.01)
    plt.show()
    return res if len(nicl_numbers) > 1 else res[0]


def create_poly_file_for_osm_clipping(outfile, buff=100):
    camden = get_camden_region().buffer(buff)
    camden.transform(4326)
    with open(outfile, 'w') as f:
        f.write('%s\n' % outfile)
        f.write('1\n')
        for lon, lat in camden.coords[0]:
            f.write('%f %f\n' % (lon, lat))
        f.write('END\n')
        f.write('END\n')
    ## Now run: osmosis --read-xml file='unclipped.osm' --bounding-polygon completeWays=yes file='outfile'
    ## --write-xml file='clipped_buffered_file.osm'
