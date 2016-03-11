from geos import plot_shapely_geos
from shapely import ops
from matplotlib import pyplot as plt
import bisect
import numpy as np


def plot_plain_network_edges(lines, buff=10., ax=None, **plot_kwargs):
    """
    Plot the linestrings in the supplied iterable by buffering and collapsing to
    an mpoly. This avoids using descartes
    TODO: broken due to overly simplistic plot_shapely_geos. Need descartes.
    """
    ax = ax or plt.gca()
    polys = [l.buffer(buff) for l in lines]
    combined = ops.cascaded_union(polys)
    
    if not plot_kwargs:
        plot_kwargs = {'color': 'k', 'lw': 1.}
    
    plot_shapely_geos(combined, ax=ax, **plot_kwargs)
    

def scatterplot_network_values(net_sample_points, values, ax=None, cmap='Reds', fmax=None, vmax=None, s=20., alpha=None):
    """
    :param s: Markersize for scatter points
    :param fmax: Maximum value as proportion of CDF
    :param vmax: Absolute maximum value to plot
    """
    ax = ax or plt.gca()
    
    values = np.array(values)
    if fmax and vmax:
        raise AttributeError("Supply fmax OR vmax, not both")
            
    if fmax:
        idx = bisect.bisect_left(np.linspace(0, 1, values.size), fmax)
        vmax = sorted(values)[idx]
        
    if vmax:
        values[values > vmax] = vmax
        
    xy = net_sample_points.to_cartesian()
    ax.scatter(xy.toarray(0), xy.toarray(1), c=values, cmap=cmap, s=s, edgecolor='none', alpha=alpha)

    