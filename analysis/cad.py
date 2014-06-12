__author__ = 'gabriel'
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
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
from point_process import runner, estimation

UK_TZ = pytz.timezone('Europe/London')

mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['interactive'] = True


def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=dt.tzinfo)
    delta = dt - epoch
    return delta.total_seconds()


class CadAggregate(object):
    def __init__(self, nicl_number=None, only_new=False, dedupe=True):
        self.nicl_number = nicl_number
        if nicl_number:
            self.nicl_name = models.Nicl.objects.get(number=nicl_number).description
        else:
            self.nicl_name = 'All crime types'
        self.only_new = only_new
        self.dedupe = dedupe
        self.cad = None
        self.load_data()

    def load_data(self):
        self.cad = logic.initial_filter_cad(nicl_type=self.nicl_number, only_new=self.only_new)
        if self.dedupe:
            self.cad = logic.dedupe_cad(self.cad)

    @property
    def start_date(self):
        return min([x.inc_datetime for x in self.cad]).replace(hour=0, minute=0, second=0)

    @property
    def end_date(self):
        return max([x.inc_datetime for x in self.cad])

    def aggregate(self):
        raise NotImplementedError()


class CadSpatialGrid(CadAggregate):
    def __init__(self, nicl_number=None, grid=None, only_new=False, dedupe=True):
        # defer dedupe until after spatial aggregation
        super(CadSpatialGrid, self).__init__(nicl_number=nicl_number, only_new=only_new, dedupe=False)
        self.dedupe = dedupe
        # load grid or use one provided
        self.grid = grid or models.Division.objects.filter(type='cad_250m_grid')
        self.shapely_grid = pandas.Series([geodjango_to_shapely([x.mpoly]) for x in self.grid],
                                          index=[x.name for x in self.grid])
        # gridded data
        self.data = self.aggregate()

    def aggregate(self):
        return logic.cad_aggregate_grid(self.cad, grid=self.grid, dedupe=self.dedupe)


class CadTemporalAggregation(CadAggregate):
    def __init__(self, nicl_number=None, only_new=False, dedupe=True):
        super(CadTemporalAggregation, self).__init__(nicl_number=nicl_number, only_new=only_new, dedupe=dedupe)
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


def apply_point_process_to_cad(nicl_type=3):
    max_trigger_t = 30 # units days
    max_trigger_d = 500 # units metres
    # min_bandwidth = np.array([0., 125., 125.])
    min_bandwidth = None
    qset = logic.clean_dedupe_cad(nicl_type=nicl_type, only_new=False)
    xy = np.array([x.att_map.coords for x in qset])
    new_xy = jiggle_on_grid_points(xy[:, 0], xy[:, 1])
    rel_dt = np.min([x.inc_datetime for x in qset])
    t = np.array([[(x.inc_datetime - rel_dt).total_seconds() / float(24 * 60 * 60)] for x in qset])
    res = np.hstack((t, new_xy))
    # sort data
    res = res[np.argsort(res[:, 0])]
    # manually estimate p initially
    p = estimation.initial_guess_educated(res, ct=1, cd=0.02)
    r = runner.PointProcess(res, p=p, max_trigger_t=max_trigger_t, max_trigger_d=max_trigger_d,
                            min_bandwidth=min_bandwidth)
    r.train(niter=50)
    return r


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



class CadByGrid(object):

    def __init__(self, nicl_numbers=range(1, 16), grid=None):
        self.nicl_numbers = nicl_numbers
        self.nicl_names = [models.Nicl.objects.get(number=x).description for x in nicl_numbers]
        self.grid = grid or models.Division.objects.filter(type='cad_250m_grid')
        self.shapely_grid = pandas.Series([geodjango_to_shapely([x.mpoly]) for x in self.grid],
                                          index=[x.name for x in self.grid])
        # preliminary cad filter
        self.cad = logic.initial_filter_cad()

        self.res, self.start_date, self.end_date = self.compute_array()

    @property
    def l(self):
        return len(self.nicl_numbers)

    @property
    def m(self):
        return self.grid.count()

    def compute_array(self):
        res = [[[] for y in range(self.m)] for x in range(self.l)]

        start_date = datetime.datetime.now(tz=UK_TZ)
        end_date = datetime.datetime(1990, 1, 1, tzinfo=UK_TZ)

        for i in range(self.l):
            nicl = self.nicl_numbers[i]
            # filter by crime type and de-dupe
            this_qset = self.cad.filter(Q(cl01=nicl) | Q(cl02=nicl) | Q(cl03=nicl)).values(
                'att_map',
                'cris_entry',
                'inc_datetime',
                ).distinct('cris_entry')
            for j in range(self.m):
                this_grid = self.grid[j]
                qry = {'att_map__within': this_grid.mpoly}
                res[i][j] = [x['inc_datetime'] for x in this_qset.filter(**qry)]
                if len(res[i][j]):
                    start_date = min(start_date, min(res[i][j]))
                    end_date = max(end_date, max(res[i][j]))

        return res, start_date, end_date

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

    def time_aggregate_data(self, bucket_dict):
        index = [x.name for x in self.grid]
        columns = self.nicl_names
        n = len(bucket_dict)

        data = np.zeros((n, self.m, self.l))
        for i in range(self.l): # crime types
            for j in range(self.m): # grid squares
                for k, func in enumerate(bucket_dict.values()): # time buckets
                    data[k, j, i] = len([x for x in self.res[i][j] if func(x)])

        if n == 1:
            data = np.squeeze(data, axis=(0,))
            return pandas.DataFrame(data, index=index, columns=columns)
        else:
            return pandas.Panel(data, items=bucket_dict.keys(), major_axis=index, minor_axis=columns)


    ## TODO: add methods for pivoting the data, aggregating by time, etc
    ## TODO: look into using ragged DataFrame?


def global_i_analysis():

    short_names = ['Violence', 'Sexual Offences', 'Burglary Dwelling', 'Burglary Non-dwelling',
                   'Robbery', 'Theft of Vehicle', 'Theft from Vehicle', 'Other Theft',
                   'Fraud and Forgery', 'Criminal Damage', 'Drug Offences', 'Bomb Threat',
                   'Shoplifting', 'Harassment', 'Abduction/Kidnap']

    cbg = CadByGrid()
    a = cbg.all_time_aggregate()
    W = logic.rook_boolean_connectivity(cbg.grid)
    global_i = [(x, logic.global_morans_i_p(a[x], W, n_iter=5000)) for x in a]

    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.75])
    hbar = ax.bar(range(cbg.l), [x[1][0] for x in global_i], width = 0.8)
    for i in range(cbg.l):
        if global_i[i][1][1] < 0.01:
            hbar[i].set_color('r')
        elif global_i[i][1][1] < 0.05:
            hbar[i].set_color('y')
        else:
            hbar[i].set_color('b')
    ax.set_xticks([float(x) + 0.5 for x in range(cbg.l)])
    ax.set_xticklabels(short_names)
    ax.set_ylabel('Global Moran''s I')
    ax.set_xlabel('Crime type (NICL)')

    xticks = ax.xaxis.get_ticklabels()
    plt.setp(xticks, rotation=90)
    plt.show()


def numbers_by_type():

    short_names = ['Violence', 'Sexual Offences', 'Burglary Dwelling', 'Burglary Non-dwelling',
                   'Robbery', 'Theft of Vehicle', 'Theft from Vehicle', 'Other Theft',
                   'Fraud and Forgery', 'Criminal Damage', 'Drug Offences', 'Bomb Threat',
                   'Shoplifting', 'Harassment', 'Abduction/Kidnap']

    cbg = CadByGrid()
    a = cbg.all_time_aggregate()
    num_crimes = a.sum()

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.75])
    hbar = ax.bar(range(cbg.l), num_crimes.values, width=0.8)

    ax.set_xticks([float(x) + 0.5 for x in range(cbg.l)])
    ax.set_xticklabels(short_names)
    ax.set_ylabel('Number in 1 year')
    ax.set_xlabel('Crime type (NICL)')

    xticks = ax.xaxis.get_ticklabels()
    plt.setp(xticks, rotation=90)
    plt.show()


def spatial_density_all_time():

    nicl_numbers = [3, 6, 10]
    short_names = ['Burglary Dwelling', 'Veh Theft', 'Crim Damage']
    camden_mpoly = geodjango_to_shapely([models.Division.objects.get(name='Camden', type='borough').mpoly])
    cbg = CadByGrid(nicl_numbers=nicl_numbers)
    a = cbg.all_time_aggregate()

    fig = plt.figure(figsize=(15, 6))
    axes = [fig.add_subplot(1, cbg.l, i+1, projection=ccrs.OSGB()) for i in range(cbg.l)]
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.03, hspace=0.01)

    for i in range(cbg.l):
        ax = axes[i]
        ds = a[cbg.nicl_names[i]]
        ax.set_title(cbg.nicl_names[i])
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
    nicl_numbers = [3, 6, 10]
    short_names = ['Burglary Dwelling', 'Veh Theft', 'Crim Damage']
    camden_mpoly = geodjango_to_shapely([models.Division.objects.get(name='Camden', type='borough').mpoly])
    cbg = CadByGrid(nicl_numbers=nicl_numbers)
    a = cbg.all_time_aggregate()
    W = logic.rook_boolean_connectivity(cbg.grid)

    fig = plt.figure(figsize=(15, 6))
    axes = [fig.add_subplot(1, cbg.l, i+1, projection=ccrs.OSGB()) for i in range(cbg.l)]
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.03, hspace=0.01)

    for i in range(cbg.l):
        ax = axes[i]
        ds = a[cbg.nicl_names[i]]
        local_i = logic.local_morans_i(ds, W)

        ax.set_title(cbg.nicl_names[i])
        ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
        ax.background_patch.set_visible(False)
        # ax.outline_patch.set_visible(False)
        cmap = mpl.cm.cool
        norm = mpl.colors.Normalize()
        norm.autoscale(local_i)
        cax = mpl.colorbar.make_axes(ax, location='bottom', pad=0.02, fraction=0.05, shrink=0.9)
        cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for j in range(cbg.m):
            val = local_i.values[j]
            fc = sm.to_rgba(val) if val else 'none'
            ax.add_geometries(geodjango_to_shapely([cbg.grid[j].mpoly]), ccrs.OSGB(), facecolor=fc)

        ax.add_geometries(camden_mpoly, ccrs.OSGB(), facecolor='none', edgecolor='black')


    plt.show()

def something_else():

    nicl_cat = {
        'Burglary Dwelling': models.Nicl.objects.get(number=3),
        'Violence Against The Person': models.Nicl.objects.get(number=1),
        'Shoplifting': models.Nicl.objects.get(number=13),
    }

    grid = models.Division.objects.filter(type='cad_250m_grid')
    shapely_grid = [geodjango_to_shapely([x.mpoly]) for x in grid]

    cad_qset = models.Cad.objects.exclude(cris_entry__isnull=True).exclude(cris_entry__startswith='NOT').exclude(att_map__isnull=True)
    res_all = collections.OrderedDict()
    res_weekly = []
    camden_mpoly = geodjango_to_shapely([models.Division.objects.get(name='Camden', type='borough').mpoly])

    cad_sections = {}

    l = len(nicl_cat)
    m = grid.count()

    res = [[[] for y in range(m)] for x in range(l)]
    start_date = datetime.datetime.now(tz=UK_TZ)
    end_date = datetime.datetime(1990, 1, 1, tzinfo=UK_TZ)

    for i in range(l):
        nicl = nicl_cat.values()[i]
        this_qset = cad_qset.filter(Q(cl01=nicl) | Q(cl02=nicl) | Q(cl03=nicl)).values(
            'att_map',
            'cris_entry',
            'inc_datetime',
            ).distinct('cris_entry')
        for j in range(m):
            this_grid = grid[j]
            res[i][j] = [x['inc_datetime'] for x in this_qset.filter(att_map__within=this_grid.mpoly)]
            if len(res[i][j]):
                start_date = min(start_date, min(res[i][j]))
                end_date = max(end_date, max(res[i][j]))

    # aggregate over all time
    all_time = np.zeros((l, m))
    for i in range(l):
        for j in range(m):
            all_time[i, j] += len(res[i][j])

    # aggregate monthly
    n = len(list(month_iterator(start_date, end_date)))
    monthly = np.zeros((l, m, n))
    start_date = start_date.replace(day=1, hour=0, minute=0, second=0)

    for i in range(l):
        for j in range(m):
            m_it = month_iterator(start_date, end_date)
            for k in range(n):
                sd, ed = m_it.next()
                monthly[i, j, k] += len([x for x in res[i][j] if sd <= x < ed])

    # aggregate weekday/weekend
    weekday = np.zeros((l, m, 2))

    for i in range(l):
        for j in range(m):
            weekday[i, j, 0] = len([x for x in res[i][j] if x.weekday() < 5])
            weekday[i, j, 1] = len([x for x in res[i][j] if x.weekday() >= 5])

    # aggregate daytime/evening+night
    timeofday = np.zeros((l, m, 2))

    am = datetime.time(6, 0, 0, tzinfo=UK_TZ)
    pm = datetime.time(18, 0, 0, tzinfo=UK_TZ)
    for i in range(l):
        for j in range(m):
            timeofday[i, j, 0] = len([x for x in res[i][j] if am <= x.time() < pm])
            timeofday[i, j, 1] = len([x for x in res[i][j] if (pm <= x.time()) or (x.time() < am)])


## so SLOW:
# def pairwise_distance(nicl_number=3):
#     cad = initial_filter_cad()
#     cad = cad.filter(Q(cl01=nicl_number) | Q(cl02=nicl_number) | Q(cl03=nicl_number)).distinct('cris_entry')
#     n = cad.count()
#     p = np.zeros((n, n))
#     for i in range(n):
#         a = cad.distance(cad[i].att_map, field_name='att_map')
#         p[i, i+1:] = np.array([x.distance.m for x in a[i+1:]])
#         if i % 100 == 0:
#             print i
#     return list(cad), p


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
    from itertools import combinations
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


def pairwise_time_lag_events(max_distance=200, nicl_numbers=None, num_bins=None):
    """ Recreate Fig 1(b) Mohler et al 2011 'Self-exciting point process modeling of crime'
        max_distance is in units of metres. """

    nicl_numbers = nicl_numbers or [1, 3, 13]
    n = len(nicl_numbers)
    res = []
    cad = logic.initial_filter_cad(only_new=True)
    for nicl in nicl_numbers:
        pd = pairwise_distance(nicl)
        pt = pairwise_time_difference(nicl)
        filt_sub = np.where(pd < max_distance)[0]
        time_diffs = pt[filt_sub]
        res.append(time_diffs)

    fig = plt.figure()
    for i in range(n):
        ax = fig.add_subplot(1, n, i)
        n_win = max(res[i])
        D = 0.5 * n_win * (n_win - 1)
        ax.hist(res[i], num_bins or range(n_win), normed=True, edgecolor='none', facecolor='gray')
        ax.plot(np.arange(1, n_win), np.arange(1, n_win)[::-1]/D, 'k--')
        ax.set_xlabel('Time difference (days)')
        ax.set_ylabel('Event pair density')

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, wspace=0.03, hspace=0.01)
    plt.show()
    return res if len(nicl_numbers) > 1 else res[0]


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
