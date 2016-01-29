__author__ = 'gabriel'
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib import colors
import numpy as np


def centre_axes(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.spines['left'].set_position(('data', 0.))
    ax.spines['bottom'].set_position(('data', 0.))


def custom_colourmap_white_to_colour(col, name='custom', reverse=False):
    """
    Create a custom colourmap that goes from white to the specified colour,
    in a similar way to Reds
    """
    cols = [
        (1, 1, 1),
        col
    ]
    if reverse:
        cols = cols[::-1]
    return mpl.colors.LinearSegmentedColormap.from_list(name, cols)


def colour_mapper(data,
                  vmin=None,
                  vmax=None,
                  fmin=None,
                  fmax=None,
                  cmap=cm.get_cmap('Reds')):

    data = np.array(data)

    if fmin:
        vmin = sorted(data)[int(data.size * fmin)]
    elif vmin is None:
        vmin = data.min()

    if fmax:
        vmax = sorted(data)[int(data.size * fmax)]
    elif vmax is None:
        vmax = data.max()

    cmap = cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def transparent_colour_map(cmap='Reds', reverse=False):
    """
    Add alpha data to the supplied colour map. Alpha is ramped linearly from 0 -> 1
    :param cmap: String or cm instance
    :param reverse: If True, alpha channel has opposite direction
    :return: LinearSegmentedColormap
    """
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    data = cmap.__dict__['_segmentdata']
    n = len(data['red'])
    if reverse:
        alphas = np.linspace(0., 1., n)[::-1]
    else:
        alphas = np.linspace(0., 1., n)
    data['alpha'] = [(i, i, i) for i in alphas]
    return colors.LinearSegmentedColormap("%s_t" % cmap.name, data)


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
    
    
def abs_bound_from_rel(arr, fracs):
    """
    Given the arbitrarily-shaped and ordered array arr, compute the bound
    or bounds corresponding to the proportional rank(s) in fracs
    """
    if not hasattr(fracs, '__iter__'):
        fracs = [fracs]
    sorted_arr = np.array(arr).flatten()
    sorted_arr.sort()
    res = []
    for f in fracs:
        assert 0 <= f <= 1, 'All supplied fracs must be in the range [0, 1]'
        if f == 1:
            f -= 1e-16
        res.append(sorted_arr[int(f * sorted_arr.size)])
    if len(res) == 1:
        res = res[0]
    return res