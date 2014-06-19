__author__ = 'gabriel'
from database import models
from shapely import geometry as shapely_geometry
from django.contrib.gis import geos
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from descartes import PolygonPatch
import json


def geodjango_to_shapely(x, c=ccrs.OSGB()):
    """ Convert geodjango geometry to shapely for plotting etc
        inputs: x is a sequence of geodjango geometry objects """

    polys = []
    for t in x:
        if isinstance(t, geos.Polygon):
            polys.append(shapely_geometry.Polygon(t.coords))
        elif isinstance(t, geos.MultiPolygon):
            polys.append(shapely_geometry.MultiPolygon([shapely_geometry.Polygon(x[0]) for x in t.coords]))

    return polys


def polygonpatch_from_polygon(poly):
    return PolygonPatch(json.loads(poly.geojson))

def plot_geodjango_shapes(shapes, ax=None):
    # shapes is an iterable containing Geodjango GEOS objects
    # returns plot objects

    ax = ax or plt.gca()
    res = []

    for s in shapes:
        if isinstance(s, geos.Point):
            res.append(ax.plot(s.coords[0], s.coords[1], 'ko'))
        elif isinstance(s, geos.LineString):
            ##TODO: implement
            raise NotImplementedError()
        elif isinstance(s, geos.Polygon):
            res.append(ax.add_patch(polygonpatch_from_polygon(s)))
        elif isinstance(s, geos.MultiPolygon):
            this_res = []
            for poly in s:
                this_res.append(ax.add_patch(polygonpatch_from_polygon(poly)))
            res.append(this_res)

    return res