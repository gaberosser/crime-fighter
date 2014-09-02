__author__ = 'gabriel'
import numpy as np
from shapely import geometry as shapely_geometry
from django.contrib.gis import geos
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib import cm
from descartes import PolygonPatch
import json
from analysis.spatial import is_clockwise


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
            res.append(ax.plot(s.coords[0], s.coords[1], 'ko', **kwargs))
        elif isinstance(s, geos.LineString):
            lsc = s.coords
            x = [t[0] for t in s.coords]
            y = [t[1] for t in s.coords]
            res.append(ax.plot(x, y, **kwargs))
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


def plot_surface_on_polygon(poly, func, n=50, cmap=cm.jet, nlevels=10,
                            vmax=None, fmax=None, egrid=None):
    """
    :param poly: geos Polygon or Multipolygon defining region
    :param func: function accepting two vectorized input arrays returning the values to be plotted
    :param n: number of pts along one side (approx)
    :param cmap: matplotlib cmap to use
    :param egrid: egrid member of RocSpatial for plotting.  No grid is plotted if None.
    :param vmax: maximum value to assign on colourmap - values beyond this are clipped
    :param fmax: maximum value on CDF at which to clip z values
    :return:
    """
    if fmax and vmax:
        raise AttributeError("Either specify vmax OR fmax")

    x_min, y_min, x_max, y_max = poly.extent
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xx, yy = np.meshgrid(x, y, copy=False)
    zz = func(xx, yy)

    if vmax:
        zz[zz > vmax] = vmax

    if fmax:
        tmp = sorted(zz.flat)
        cut = int(np.floor(len(tmp) * fmax))
        vmax = tmp[cut]
        zz[zz > vmax] = vmax

    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = plt.contourf(xx, yy, zz, nlevels)

    # plot grid if required
    if egrid is not None:
        egrid = np.array(egrid)
        xu = np.unique(np.vstack((egrid[:, 0], egrid[:, 2])))
        yu = np.unique(np.vstack((egrid[:, 1], egrid[:, 3])))
        for x in xu:
            ax.plot(np.ones(2) * x, [y_min, y_max], 'w-', alpha=0.3)
        for y in yu:
            ax.plot([x_min, x_max], np.ones(2) * y, 'w-', alpha=0.3)

    poly_verts = list(poly.coords[0])
    # check handedness of poly
    if is_clockwise(poly):
        poly_verts = poly_verts[::-1]

    mask_outside_polygon(poly_verts, ax=ax)
    plot_geodjango_shapes((poly,), ax=ax, facecolor='none')

    return h


def mask_outside_polygon(poly_verts, ax=None):
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.

    "poly_verts" must be a list of tuples of the vertices in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    if ax is None:
        ax = plt.gca()

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Verticies of the plot boundaries in clockwise order
    bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]),
                   (xlim[1], ylim[1]), (xlim[1], ylim[0]),
                   (xlim[0], ylim[0])]

    # A series of codes (1 and 2) to tell matplotlib whether to draw a line or
    # move the "pen" (So that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

    # Plot the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none')
    patch = ax.add_patch(patch)

    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return patch