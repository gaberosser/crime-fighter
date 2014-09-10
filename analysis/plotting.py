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
    # shapes is one or an iterable of Geodjango GEOS objects
    # returns plot object(s)

    ax = ax or plt.gca()
    res = []
    x_min = y_min = 1e8
    x_max = y_max = -1e8

    if issubclass(shapes.__class__, geos.GEOSGeometry):
        # single GEOS geometry supplied
        shapes = [shapes]

    pts = []

    for s in shapes:
        if set_axes:
            x_min = min(x_min, s.extent[0])
            y_min = min(y_min, s.extent[1])
            x_max = max(x_max, s.extent[2])
            y_max = max(y_max, s.extent[3])
        if isinstance(s, geos.Point):
            pts.append(s.coords)
            # res.append(ax.plot(s.coords[0], s.coords[1], 'ko', **kwargs))
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

    # plot all points together
    if len(pts):
        pts = np.array(pts)
        res.append(ax.plot(pts[:, 0], pts[:, 1], 'ko', **kwargs))

    if set_axes:
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim([x_min - x_range * 0.02, x_max + x_range * 0.02])
        ax.set_ylim([y_min - y_range * 0.02, y_max + y_range * 0.02])
        ax.set_aspect('equal')
    return res


def plot_surface_on_polygon(poly, func, ax=None, n=50, cmap=cm.jet, nlevels=50,
                            vmin=None, vmax=None, fmax=None, egrid=None, **kwargs):
    """
    :param poly: geos Polygon or Multipolygon defining region
    :param func: function accepting two vectorized input arrays returning the values to be plotted
    :param n: number of pts along one side (approx)
    :param cmap: matplotlib cmap to use
    :param nlevels: number of contour colour levels to use
    :param egrid: egrid member of RocSpatial for plotting.  No grid is plotted if None.
    :param vmin: minimum value to plot. Values below this are left unfilled
    :param vmax: maximum value to assign on colourmap - values beyond this are clipped
    :param fmax: maximum value on CDF at which to clip z values
    :param kwargs: any other kwargs are passed to the plt.contourf call
    :return:
    """
    if fmax and vmax:
        raise AttributeError("Either specify vmax OR fmax")

    x_min, y_min, x_max, y_max = poly.extent
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xx, yy = np.meshgrid(x, y, copy=False)
    zz = func(xx, yy)

    if vmax is None:
        vmax = np.max(zz)

    if vmin is None:
        vmin = np.min(zz)

    if fmax:
        tmp = sorted(zz.flat)
        cut = int(np.floor(len(tmp) * fmax))
        vmax = tmp[cut]
        # zz[zz > vmax] = vmax

    # clip max values to vmax so they still get drawn:
    zz[zz > vmax] = vmax

    levels = np.linspace(vmin, vmax, nlevels)

    if not ax:
        fig = plt.figure()
        buf = 2e-2
        ax = fig.add_axes([buf, buf, 1 - 2 * buf, 1 - 2 * buf])
        ax.axis('off')

    cont = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, **kwargs)

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

    # mask_outside_polygon(poly_verts, ax=ax)
    mask_contour(cont, poly_verts, ax=ax, show_clip_path=True)
    plot_geodjango_shapes(poly, ax=ax, facecolor='none')

    return cont


def mask_outside_polygon(poly_verts, ax=None):
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.

    "poly_verts" must be a list of tuples of the vertices in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    """

    if ax is None:
        ax = plt.gca()

    # fraction by which to extend outer bound
    # required to avoid slithers of underlying surface sticking out
    buf_frac = 0.01

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xbuf = np.diff(xlim)[0] * buf_frac
    ybuf = np.diff(ylim)[0] * buf_frac


    # Verticies of the plot boundaries in clockwise order
    bound_verts = [
        (xlim[0] - xbuf, ylim[0] - ybuf),
        (xlim[0] - xbuf, ylim[1] + ybuf),
        (xlim[1] + xbuf, ylim[1] + ybuf),
        (xlim[1] + xbuf, ylim[0] - ybuf),
        (xlim[0] - xbuf, ylim[0] - ybuf)
    ]

    # A series of codes (1 and 2) to tell matplotlib whether to draw a line or
    # move the "pen" (So that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    # poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

    # can also implement this with a closing statement at the end:
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 2) * [mpath.Path.LINETO] + [mpath.Path.CLOSEPOLY]

    # Create the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none')

    # apply the masking patch
    patch = ax.add_patch(patch)

    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return patch


def mask_contour(cont, poly_verts, ax=None, show_clip_path=True):

    if ax is None:
        ax = plt.gca()

    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 2) * [mpath.Path.LINETO] + [mpath.Path.CLOSEPOLY]
    path = mpath.Path(poly_verts, poly_codes)
    if show_clip_path:
        ec = 'k'
    else:
        ec = 'none'
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=ec)

    ax.add_patch(patch)

    for col in cont.collections:
        col.set_clip_path(patch)

    plt.draw()
    return patch