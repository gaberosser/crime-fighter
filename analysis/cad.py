__author__ = 'gabriel'
from database import models
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import collections
from django.db.models import Q, Count, Sum, Min, Max
from plotting import geodjango_to_shapely
from database.views import month_iterator, week_iterator

mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['interactive'] = True

# def spatial_cad():
#

nicl_cat = {
    'Burglary Dwelling': models.Nicl.objects.get(number=3),
    'Violence Against The Person': models.Nicl.objects.get(number=1),
    'Shoplifting': models.Nicl.objects.get(number=13),
}

grid = models.Division.objects.filter(type='cad_250m_grid')
cad_qset = models.Cad.objects.exclude(cris_entry__isnull=True).exclude(cris_entry__startswith='NOT').exclude(att_map__isnull=True)
res_all = collections.OrderedDict()
res_weekly = []
camden_mpoly = geodjango_to_shapely([models.Division.objects.get(name='Camden', type='borough').mpoly])

cad_sections = {}

# separate by category
for k, v in nicl_cat.items():
    cad_sections[k] = cad_qset.filter(Q(cl01=v) | Q(cl02=v) | Q(cl03=v)).values(
        'att_map',
        'cris_entry',
        'inc_datetime',
        ).distinct('cris_entry')

all_dates = []
for v in cad_sections.values():
    all_dates += [x['inc_datetime'] for x in v]
start_date = min(all_dates).date()
end_date = max(all_dates).date()

# aggregate by grid squares
cad_grid = {}
for cg in grid:
    cad_grid[cg.name] = dict([
        (k, v.filter(att_map__within=cg.mpoly)) for k, v in cad_sections.items()
    ])
    cad_grid[cg.name]['mpoly'] = geodjango_to_shapely([cg.mpoly])

# iterate over grid squares and aggregate over all time
for k, v in cad_grid.items():
    this_dict = {}
    for c in nicl_cat.keys():
        this_dict[c] = v[c].count()
    this_dict['mpoly'] = v['mpoly']
    res_all[k] = this_dict

# aggregate by month
## TODO: slow due to filter in triple loop, rearrange
res_monthly = {}
for gridname, cg in cad_grid.items():
    this_dict = {}
    for k in nicl_cat.keys():
        this_dict2 = {}
        for (sd, ed) in month_iterator(start_date, end_date):
            this_dict2[sd] = cg[k].filter(inc_datetime__gte=sd, inc_datetime__lt=ed).count()
        this_dict[k] = this_dict2
    res_monthly[gridname] = this_dict

fig = plt.figure()
fig.set_size_inches(12, 9)

for i, k in enumerate(nicl_cat.keys()):

    ax = fig.add_subplot(1, 3, i+1, projection=ccrs.OSGB())
    ax.set_title(k)
    ax.set_extent([523000, 533000, 179000, 190000], ccrs.OSGB())
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize()
    norm.autoscale([x[k] for x in res_all.values()])
    cax = mpl.colorbar.make_axes(ax, location='bottom')
    cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for x in res_all.values():
        val = x[k]
        fc = sm.to_rgba(val) if val else 'none'
        ax.add_geometries(x['mpoly'], ccrs.OSGB(), facecolor=fc)

    ax.add_geometries(camden_mpoly, ccrs.OSGB(), facecolor='none', edgecolor='black')

plt.show()