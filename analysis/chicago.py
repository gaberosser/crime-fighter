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
from point_process import estimation, models as pp_models
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


def apply_point_process_to_chicago():

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