__author__ = 'gabriel'
import numpy as np
from utils import mask_contour
from shapely import geometry as shapely_geometry
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from descartes import PolygonPatch
import json
from analysis.spatial import is_clockwise, bounding_box_grid, geodjango_to_shapely

try:
    from django.contrib.gis import geos
    HAS_GEODJANGO = True
except ImportError:
    geos = None
    HAS_GEODJANGO = False


def polygonpatch_from_geodjango_polygon(poly, **kwargs):
    return PolygonPatch(json.loads(poly.geojson), **kwargs)


def plot_geodjango_shapes(shapes, ax=None, set_axes=True, **kwargs):
    # shapes is one or an iterable of Geodjango GEOS objects
    # returns plot object(s)
    assert HAS_GEODJANGO, "Cannot plot geodjango shapes without geodjango"

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
            res.append(ax.add_patch(polygonpatch_from_geodjango_polygon(s, **kwargs)))
        elif isinstance(s, geos.MultiPolygon):
            this_res = []
            for poly in s:
                this_res.append(ax.add_patch(polygonpatch_from_geodjango_polygon(poly, **kwargs)))
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
        shapely_geometry.GeometryCollection:
            lambda s, ax, **kwargs: [plot_shapely_geos(t, ax, **kwargs) for t in s],
    }

    if issubclass(shapes.__class__, shapely_geometry.base.BaseGeometry):
        # single GEOS geometry supplied
        return plotters[shapes.__class__](shapes, ax, **kwargs)

    ## TODO: may wish to combine plotting of points for efficiency?
    return [plotters[t.__class__](t, ax, **kwargs) for t in shapes]


def plot_shaded_regions(polys,
                        values,
                        domain=None,
                        ax=None,
                        cmap=cm.jet,
                        vmin=None,
                        vmax=None,
                        fmax=None,
                        alpha=None,
                        colorbar=True,
                        scale_bar=1000,
                        scale_label='1 km',
                        scale_bar_loc='se'):
    """
    :param polys:
    :param values:
    :param domain:
    :param ax:
    :param cmap:
    :param vmin:
    :param vmax:
    :param fmax:
    :param alpha:
    :param colorbar:
    :param scale_bar: Scale bar length in same units as polys , or None to disable
    :return:
    """

    if fmax and vmax:
        raise AttributeError("Either specify vmax OR fmax")

    if len(polys) != len(values):
        raise AttributeError("polys and values must be the same size")

    ax = ax or plt.gca()

    for i in range(len(polys)):
        if isinstance(polys[i], geos.GEOSGeometry):
            polys[i] = geodjango_to_shapely(polys[i])

    if vmax is None:
        vmax = np.max(values)

    if vmin is None:
        vmin = np.min(values)

    if fmax:
        tmp = sorted(values)
        cut = int(np.floor(len(tmp) * fmax))
        vmax = tmp[cut]

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for p,v in zip(polys, values):
        if v < vmin:
            continue
        plot_shapely_geos(p, ax=ax, fc=sm.to_rgba(v), ec='black')

    if colorbar:
        cax = mpl.colorbar.make_axes(ax, location='bottom', pad=0.02, fraction=0.05, shrink=0.9)
        cbar = mpl.colorbar.ColorbarBase(cax[0], cmap=cmap, norm=norm, orientation='horizontal')

    ax.axis('off')
    ax.autoscale()
    ax.set_aspect('equal')

    if scale_bar:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        if scale_bar_loc == 'sw':
            line_y = (y0 + 0.05 * (y1 - y0)) * np.ones(2)
            line_x0 = x0 + 0.05 * (x1 - x0)
            line_x = np.array([line_x0, line_x0 + scale_bar])
        else:
            line_y = (y0 + 0.05 * (y1 - y0)) * np.ones(2)
            line_x1 = x1 - 0.05 * (x1 - x0)
            line_x = np.array([line_x1 - scale_bar, line_x1])

        ax.plot(line_x, line_y, 'k-', linewidth=4.0)
        ax.text(np.mean(line_x), line_y[0] + 1, scale_label, ha='center', va='bottom')

    if domain:
        plot_shapely_geos(domain, ax=ax, facecolor='none', ec='black')


def plot_surface_function_on_polygon(poly,
                                     func,
                                     ax=None,
                                     dx=None,
                                     offset_coords=None,
                                     cmap=cm.jet,
                                     nlevels=50,
                                     vmin=None,
                                     vmax=None,
                                     fmax=None,
                                     colorbar=False,
                                     show_domain=True,
                                     mask=True,
                                     **kwargs):
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
    :param show_domain: If True, the domain polygon is plotted
    :param mask: If True, the surface is masked using the domain polygon
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

    if mask:
        poly_verts = list(poly.exterior.coords)
        # check handedness of poly
        if is_clockwise(poly):
            poly_verts = poly_verts[::-1]

        # mask_outside_polygon(poly_verts, ax=ax)
        mask_contour(cont, poly_verts, ax=ax, show_clip_path=show_domain)

    elif show_domain:
        plot_shapely_geos(poly, ax=ax)

    plt.axis('equal')
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