__author__ = 'gabriel'
import warnings
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
from point_process import estimation, models as pp_models, validate
from analysis import validation, hotspot
import settings
import os
from django.contrib.gis.gdal import DataSource

CHICAGO_DATA_DIR = os.path.join(settings.DATA_DIR, 'chicago')
SRID = 2028

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


def get_crimes_by_type(crime_type='burglary', **filter_dict):

    res = models.Chicago.objects.filter(primary_type__icontains=crime_type, **filter_dict).transform(SRID)
    t = [x.datetime for x in res]
    t0 = min(t)
    t = np.array([(x - t0).total_seconds() / float(60 * 60 * 24) for x in t]).reshape((len(t), 1))
    xy = np.array([x.location.coords for x in res])
    res = np.hstack((t, xy))
    res = res[np.argsort(res[:, 0])]
    return res, t0


def apply_point_process():

    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        datetime__gt=datetime.datetime(2010, 3, 1, 0),
        datetime__lte=datetime.datetime(2010, 6, 1, 0)
    )

    max_trigger_t = 30 # units days
    max_trigger_d = 200 # units metres
    # min_bandwidth = np.array([0., 125., 125.])
    min_bandwidth = None

    # manually estimate p initially
    p = estimation.initial_guess_educated(res, ct=1, cd=0.02)
    r = pp_models.PointProcess(p=p, max_trigger_t=max_trigger_t, max_trigger_d=max_trigger_d,
                            min_bandwidth=min_bandwidth)
    ps = r.train(data=res, niter=30, tol_p=1e-9)
    return r, ps

def validate_point_process(start_date=datetime.datetime(2001, 1, 1, 0),
                           end_date=datetime.datetime(2003, 1, 1),
                           first_training_size=50):

    if start_date + datetime.timedelta(days=first_training_size) > end_date:
        raise AttributeError("Insufficient data range")

    poly = compute_chicago_region()
    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        datetime__gt=start_date,
        datetime__lte=end_date
    )
    vb = validate.PpValidation(res, spatial_domain=poly, model_kwargs={
        'max_trigger_t': 30,
        'max_trigger_d': 200,
        'estimator': lambda x: estimation.initial_guess_educated(x, ct=1, cd=0.02),
    })
    vb.set_grid(250)
    vb.set_t_cutoff(first_training_size)
    polys, ps, carea, cfrac, pai = vb.run(dt=1, t_upper=70, niter=20)
    return polys, ps, carea, cfrac, pai

if __name__ == '__main__':

    start_date=datetime.datetime(2001, 1, 1, 0)
    end_date=datetime.datetime(2003, 1, 1)
    first_training_size = 50

    poly = compute_chicago_region()
    res, t0 = get_crimes_by_type(
        crime_type='burglary',
        datetime__gt=start_date,
        datetime__lte=end_date
    )
    vb = validate.PpValidation(res, spatial_domain=poly, model_kwargs={
        'max_trigger_t': 30,
        'max_trigger_d': 200,
        'estimator': lambda x: estimation.initial_guess_educated(x, ct=1, cd=0.02),
    })
    vb.set_grid(250)
    vb.set_t_cutoff(first_training_size)
    polys, ps, carea, cfrac, pai = vb.run(dt=1, niter=20, t_upper=first_training_size + 20)

    # use basic historic data spatial hotspot
    sk7 = hotspot.SKernelHistoric(2) # use heatmap from final 7 days data
    vb_sk7 = validation.ValidationBase(res, hotspot.Hotspot, poly, model_args=(sk7,))
    vb_sk7.set_grid(grid_length=250)
    vb_sk7.set_t_cutoff(first_training_size)
    polys_sk7, ps_sk7, carea_sk7, cfrac_sk7, pai_sk7 = vb_sk7.run(dt=1, niter=20, t_upper=first_training_size + 20)