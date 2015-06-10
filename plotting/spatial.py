__author__ = 'gabriel'
import numpy as np
from shapely import geometry as shapely_geometry
from django.contrib.gis import geos
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from descartes import PolygonPatch
import json
from analysis.spatial import is_clockwise, bounding_box_grid, geodjango_to_shapely


def shapely_plot(func):

    def decorator(obj, ax=None, **kwargs):
        ax = ax or plt.gca()
        return func(obj, ax, **kwargs)

    return decorator


@shapely_plot
def plot_shapely_point(obj, ax, **kwargs):
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    return ax.plot(obj.x, obj.y, **kwargs)


@shapely_plot
def plot_shapely_line(obj, ax, **kwargs):
    return ax.plot(*obj.xy, **kwargs)


@shapely_plot
def plot_shapely_polygon(obj, ax, **kwargs):
    if not ('fc' in kwargs or 'facecolor' in kwargs):
        kwargs['fc'] = 'none'
    polypatch = PolygonPatch(obj, **kwargs)
    ax.add_patch(polypatch)
    return polypatch


@shapely_plot
def plot_shapely_multipolygon(obj, ax, **kwargs):
    return [plot_shapely_polygon(t, ax, **kwargs) for t in obj]


def plot_shapely_geos(shapes, ax=None, **kwargs):

    plotters = {
        shapely_geometry.Point: plot_shapely_point,
        shapely_geometry.LineString: plot_shapely_line,
        shapely_geometry.LinearRing: plot_shapely_line,
        shapely_geometry.Polygon: plot_shapely_polygon,
        shapely_geometry.MultiPolygon: plot_shapely_multipolygon,
    }

    if issubclass(shapes.__class__, shapely_geometry.base.BaseGeometry):
        # single GEOS geometry supplied
        return plotters[shapes.__class__](shapes, ax, **kwargs)

    ## TODO: may wish to combine plotting of points for efficiency?
    return [plotters[t.__class__](t, ax, **kwargs) for t in shapes]
