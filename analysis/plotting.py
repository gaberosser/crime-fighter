__author__ = 'gabriel'
import numpy as np
from shapely import geometry as shapely_geometry
from django.contrib.gis import geos
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib import cm
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


def polygonpatch_from_polygon(poly, **kwargs):
    return PolygonPatch(json.loads(poly.geojson), **kwargs)


def plot_geodjango_shapes(shapes, ax=None, set_axes=True, **kwargs):
    # shapes is an iterable containing Geodjango GEOS objects
    # returns plot objects

    ax = ax or plt.gca()
    res = []
    x_min = y_min = 1e8
    x_max = y_max = -1e8

    for s in shapes:
        if set_axes:
            x_min = min(x_min, s.extent[0])
            y_min = min(y_min, s.extent[1])
            x_max = max(x_max, s.extent[2])
            y_max = max(y_max, s.extent[3])
        if isinstance(s, geos.Point):
            res.append(ax.plot(s.coords[0], s.coords[1], 'ko'))
        elif isinstance(s, geos.LineString):
            ##TODO: implement
            raise NotImplementedError()
        elif isinstance(s, geos.Polygon):
            res.append(ax.add_patch(polygonpatch_from_polygon(s, **kwargs)))
        elif isinstance(s, geos.MultiPolygon):
            this_res = []
            for poly in s:
                this_res.append(ax.add_patch(polygonpatch_from_polygon(poly, **kwargs)))
            res.append(this_res)

    if set_axes:
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim([x_min - x_range * 0.02, x_max + x_range * 0.02])
        ax.set_ylim([y_min - y_range * 0.02, y_max + y_range * 0.02])
    return res


def plot_surface_on_polygon(poly, func, n=50, cmap=cm.jet):
    """
    :param poly: geos Polygon or Multipolygon defining region
    :param func: function accepting two vectorized input arrays returning the values to be plotted
    :param n: number of pts along one side (approx)
    :param cmap: matplotlib cmap to use
    :return:
    """
    ex = poly.extent
    x_min = ex[0]
    y_min = ex[1]
    x_max = ex[2]
    y_max = ex[3]
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xx, yy = np.meshgrid(x, y, copy=False)
    it = np.nditer((xx, yy, None))
    for x in it:
        it[-1] = not geos.Point([x[0], x[1]]).intersects(poly)
    mask = it.operands[-1]
    zz = np.ma.array(func(xx, yy), mask=mask)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_geodjango_shapes((poly,), ax=ax, facecolor='none')
    plt.contourf(xx, yy, zz)


