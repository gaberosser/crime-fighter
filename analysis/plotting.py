__author__ = 'gabriel'
from database import models
from shapely import geometry as shapely_geometry
from django.contrib.gis.geos import Polygon, MultiPolygon
import cartopy.crs as ccrs


def geodjango_to_shapely(x, c=ccrs.OSGB()):
    """ Convert geodjango geometry to shapely for plotting etc
        inputs: x is a sequence of geodjango geometry objects """

    polys = []
    for t in x:
        if isinstance(t, Polygon):
            polys.append(shapely_geometry.Polygon(t.coords))
        elif isinstance(t, MultiPolygon):
            polys.append(shapely_geometry.MultiPolygon([shapely_geometry.Polygon(x[0]) for x in t.coords]))

    return polys