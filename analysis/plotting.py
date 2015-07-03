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


def centre_axes(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.spines['left'].set_position(('data', 0.))
    ax.spines['bottom'].set_position(('data', 0.))


def geos_poly_to_shapely(poly):
    return shapely_geometry.Polygon(shell=poly.coords[0], holes=poly.coords[1:])


def geodjango_to_shapely(shapes, c=ccrs.OSGB()):
    """ Convert geodjango geometry to shapely for plotting etc
        inputs: x is a sequence of geodjango geometry objects """

    converters = {
        geos.Point: lambda t: shapely_geometry.Point(*t.coords),
        geos.LineString: lambda t: shapely_geometry.LineString(t.coords),
        geos.Polygon: lambda t: geos_poly_to_shapely(t),
        geos.MultiPolygon: lambda t: shapely_geometry.MultiPolygon([geos_poly_to_shapely(x) for x in t])
    }

    if issubclass(shapes.__class__, geos.GEOSGeometry):
        # single GEOS geometry supplied
        return converters[shapes.__class__](shapes)

    return [converters[t.__class__](t) for t in shapes]


def polygonpatch_from_polygon(poly, **kwargs):
    return PolygonPatch(json.loads(poly.geojson), **kwargs)


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


def plot_surface_function_on_polygon(poly, func, ax=None, dx=None, offset_coords=None, cmap=cm.jet, nlevels=50,
                            vmin=None, vmax=None, fmax=None, colorbar=False, **kwargs):
    """
    :param poly: geos Polygon or Multipolygon defining region OR Shapely equivalents
    :param func: function accepting two vectorized input arrays returning the values to be plotted
    :param dx: interval distance between grid points
    :param offset_coords: iterable giving the (x, y) coordinates of a grid point, default = (0, 0)
    :param cmap: matplotlib cmap to use
    :param nlevels: number of contour colour levels to use
    :param vmin: minimum value to plot. Values below this are left unfilled
    :param vmax: maximum value to assign on colourmap - values beyond this are clipped
    :param fmax: maximum value on CDF at which to clip z values
    :param kwargs: any other kwargs are passed to the plt.contourf call
    :return:
    """
    if fmax and vmax:
        raise AttributeError("Either specify vmax OR fmax")

    if isinstance(poly, geos.GEOSGeometry):
        poly = geodjango_to_shapely(poly)

    x_min, y_min, x_max, y_max = poly.bounds

    if not dx:
        dx = ((x_max - x_min) + (y_max - y_min)) / 100

    x, y = bounding_box_grid(poly, dx, offset_coords=offset_coords)
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

    if colorbar:
        plt.colorbar(cont, shrink=0.9)

    poly_verts = list(poly.exterior.coords)
    # check handedness of poly
    if is_clockwise(poly):
        poly_verts = poly_verts[::-1]

    # mask_outside_polygon(poly_verts, ax=ax)
    mask_contour(cont, poly_verts, ax=ax, show_clip_path=True)

    # plot_geodjango_shapes(poly, ax=ax, facecolor='none')

    plt.draw()

    return xx, yy, zz


def plot_surface_on_polygon((x, y, z), poly=None, ax=None, cmap=cm.jet, nlevels=50,
                            vmin=None, vmax=None, fmax=None, colorbar=False, **kwargs):
    """
    :param poly: geos Polygon or Multipolygon defining region
    :param (x, y, z): 2D matrices holding the regularly-spaced x and y coordinates and corresponding z values
    :param cmap: matplotlib cmap to use
    :param nlevels: number of contour colour levels to use
    :param vmin: minimum value to plot. Values below this are left unfilled
    :param vmax: maximum value to assign on colourmap - values beyond this are clipped
    :param fmax: maximum value on CDF at which to clip z values
    :param kwargs: any other kwargs are passed to the plt.contourf call
    :return:
    """
    if poly and isinstance(poly, geos.MultiPolygon):
        poly = poly.simplify()
        if not isinstance(poly, geos.Polygon):
            ## TODO: if this ever becomes an issue, can probably break the multipoly into constituent parts
            raise AttributeError("Unable to use a multipolygon as a mask")


    if fmax and vmax:
        raise AttributeError("Either specify vmax OR fmax")

    if vmax is None:
        vmax = np.max(z)

    if vmin is None:
        vmin = np.min(z)

    if fmax:
        tmp = sorted(z.flat)
        cut = int(np.floor(len(tmp) * fmax))
        vmax = tmp[cut]

    # clip max values to vmax so they still get drawn:
    z[z > vmax] = vmax

    levels = np.linspace(vmin, vmax, nlevels)

    if not ax:
        fig = plt.figure()
        buf = 2e-2
        ax = fig.add_axes([buf, buf, 1 - 2 * buf, 1 - 2 * buf])
        ax.axis('equal')
        ax.axis('off')

    cont = ax.contourf(x, y, z, levels=levels, cmap=cmap, **kwargs)

    if colorbar:
        plt.colorbar(cont, shrink=0.9)

    if poly:
        poly_verts = list(poly.exterior_ring.coords)
        # check handedness of poly
        if is_clockwise(poly):
            poly_verts = poly_verts[::-1]

        # mask_outside_polygon(poly_verts, ax=ax)
        mask_contour(cont, poly_verts, ax=ax, show_clip_path=True)
        plot_shapely_geos([poly.exterior], ax=ax, facecolor='none')
        # plot_geodjango_shapes(poly, ax=ax, facecolor='none')

    plt.draw()

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