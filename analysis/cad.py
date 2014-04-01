__author__ = 'gabriel'
from database import models
from stats import logic
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial.distance import pdist, squareform
import collections
from django.db.models import Q, Count, Sum, Min, Max
from django.contrib.gis.measure import D
from plotting import geodjango_to_shapely
from database.views import month_iterator, week_iterator
import pandas
import numpy as np
import datetime
import pytz

UK_TZ = pytz.timezone('Europe/London')

mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['interactive'] = True


def initial_filter_cad():
    return models.Cad.objects.exclude(cris_entry__isnull=True).exclude(cris_entry__startswith='NOT')\
            .exclude(att_map__isnull=True)

class CadByGrid(object):

    def __init__(self, nicl_numbers=range(1, 16), grid=None):
        self.nicl_numbers = nicl_numbers
        self.nicl_names = [models.Nicl.objects.get(number=x).description for x in nicl_numbers]
        self.grid = grid or models.Division.objects.filter(type='cad_250m_grid')
        self.shapely_grid = pandas.Series([geodjango_to_shapely([x.mpoly]) for x in self.grid],
                                          index=[x.name for x in self.grid])
        # preliminary cad filter
        self.cad = initial_filter_cad()

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
    cad = initial_filter_cad()
    cad = cad.filter(Q(cl01=nicl_number) | Q(cl02=nicl_number) | Q(cl03=nicl_number)).distinct('cris_entry')
    coords = np.array([x.att_map.coords for x in cad])
    p = squareform(pdist(coords))
    return list(cad), p


def pairwise_time_lag_events(data, max_distance=D(m=200), nicl_numbers=None):
    nicl_numbers = nicl_numbers or [1, 3, 13]
    cad = initial_filter_cad()
    for nicl in nicl_numbers:
        this_cad = cad.filter(Q(cl01=nicl) | Q(cl02=nicl) | Q(cl03=nicl)).values(
                    'att_map',
                    'cris_entry',
                    'inc_datetime',
                    ).distinct('cris_entry')


