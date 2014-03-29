__author__ = 'gabriel'
from database import models
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import collections
from django.db.models import Q, Count, Sum, Min, Max
from plotting import geodjango_to_shapely
from database.views import month_iterator, week_iterator
import pandas
import numpy as np
import datetime
import pytz

UK_TZ = pytz.timezone('Europe/London')

mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['interactive'] = True


class CadByGrid(object):

    def __init__(self, nicl_numbers=range(1, 16), grid=None):
        self.nicl_numbers = nicl_numbers
        self.nicl_names = [models.Nicl.objects.get(number=x).description for x in nicl_numbers]
        self.grid = grid or models.Division.objects.filter(type='cad_250m_grid')
        self.shapely_grid = pandas.Series([geodjango_to_shapely([x.mpoly]) for x in self.grid],
                                          index=[x.name for x in self.grid])
        # preliminary cad filter
        self.cad = models.Cad.objects.exclude(cris_entry__isnull=True).exclude(cris_entry__startswith='NOT')\
            .exclude(att_map__isnull=True)

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
                res[i][j] = [x['inc_datetime'] for x in this_qset.filter(att_map__within=this_grid.mpoly)]
                if len(res[i][j]):
                    start_date = min(start_date, min(res[i][j]))
                    end_date = max(end_date, max(res[i][j]))

        return res, start_date, end_date

    def all_time_aggregate(self):
        bucket_fun = lambda x: True
        return self.time_aggregate_data({'all': bucket_fun})

    def time_aggregate_data(self, bucket_dict):
        index = self.nicl_names
        columns = [x.name for x in self.grid]
        n = len(bucket_dict)

        data = np.zeros((self.l, self.m, n))
        for i in range(self.l):
            for j in range(self.m):
                for k, func in enumerate(bucket_dict.values()):
                    data[i, j, k] = len([x for x in self.res[i][j] if func(x)])

        if n == 1:
            data = np.squeeze(data, axis=(2,))
            return pandas.DataFrame(data, index=index, columns=columns)
        else:
            return pandas.Panel(data, items=bucket_dict.keys(), major_axis=index, minor_axis=columns)


    ## TODO: add methods for pivoting the data, aggregating by time, etc
    ## TODO: look into using ragged DataFrame?


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

# fig = plt.figure()
# fig.set_size_inches(12, 9)
#
# for i in range(l):
#
#     ax = fig.add_subplot(1, 3, i+1, projection=ccrs.OSGB())
#     ax.set_title(nicl_cat.keys()[i])
#     ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
#     ax.background_patch.set_visible(False)
#     ax.outline_patch.set_visible(False)
#     cmap = mpl.cm.cool
#     norm = mpl.colors.Normalize()
#     norm.autoscale(all_time[i])
#     cax = mpl.colorbar.make_axes(ax, location='bottom')
#     cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
#     sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#
#     for j in range(m):
#         val = all_time[i, j]
#         fc = sm.to_rgba(val) if val else 'none'
#         ax.add_geometries(geodjango_to_shapely([grid[j].mpoly]), ccrs.OSGB(), facecolor=fc)
#
#     ax.add_geometries(camden_mpoly, ccrs.OSGB(), facecolor='none', edgecolor='black')
#
# plt.show()



# for i in range(l):
#     fig = plt.figure(figsize=(15, 9))
#     cmap = mpl.cm.cool
#     norm = mpl.colors.Normalize()
#     norm.autoscale(monthly[i, :, :].flatten())
#     sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#     axarr = []
#
#     for k in range(12):
#         ax = fig.add_subplot(3, 4, k+1, projection=ccrs.OSGB())
#         axarr.append(ax)
#         ax.set_title(str(k+1))
#         ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
#         ax.background_patch.set_visible(False)
#         ax.outline_patch.set_visible(False)
#         for j in range(m):
#             val = monthly[i, j, k]
#             fc = sm.to_rgba(val) if val else 'none'
#             ax.add_geometries(shapely_grid[j], ccrs.OSGB(), facecolor=fc)
#
#     fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, wspace=0.01, hspace=0.01)
#     cbar_ax = fig.add_axes([0.95, 0.1, 0.025, 0.8])
#     cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')



# cmap = mpl.cm.cool
# norm = mpl.colors.Normalize()

# for i in range(l):
#     fig = plt.figure(figsize=(15, 9))
#     for k in range(2):
#         norm.autoscale(weekday[i, :, k])
#         sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#         ax = fig.add_subplot(1, 2, k+1, projection=ccrs.OSGB())
#         ax.set_title("%s, %s" % (nicl_cat.keys()[i], "Weekday" if k == 0 else "Weekend"))
#         ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
#         ax.background_patch.set_visible(False)
#         ax.outline_patch.set_visible(False)
#         cax = mpl.colorbar.make_axes(ax, location='right')
#         cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='vertical')
#         for j in range(m):
#             val = weekday[i, j, k]
#             fc = sm.to_rgba(val) if val else 'none'
#             ax.add_geometries(shapely_grid[j], ccrs.OSGB(), facecolor=fc)
# plt.show()



# for i in range(l):
#     fig = plt.figure(figsize=(15, 9))
#     for k in range(2):
#         norm.autoscale(timeofday[i, :, k])
#         sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#         ax = fig.add_subplot(1, 2, k+1, projection=ccrs.OSGB())
#         ax.set_title("%s, %s" % (nicl_cat.keys()[i], "Daytime" if k == 0 else "Evening/night"))
#         ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
#         ax.background_patch.set_visible(False)
#         ax.outline_patch.set_visible(False)
#         cax = mpl.colorbar.make_axes(ax, location='right')
#         cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='vertical')
#         for j in range(m):
#             val = timeofday[i, j, k]
#             fc = sm.to_rgba(val) if val else 'none'
#             ax.add_geometries(shapely_grid[j], ccrs.OSGB(), facecolor=fc)
# plt.show()