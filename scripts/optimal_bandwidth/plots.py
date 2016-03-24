__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from plotting.spatial import plot_shapely_geos
from plotting.utils import colour_mapper
from shapely import ops

def plot_optimal_bandwidth_map(ht, hd, boundaries,
                               ax=None,
                               trange=None,
                               colourbar=True):
    """
    Plot a map showing both spatial and temporal optimal bandwidths
    :param ht: Dict of optimal temporal bandwidths
    :param hd: Dict of optimal spatial bandwidths
    :param boundaries: Dict of boundary polygons, indexed by same key as ht and hd
    :param ax: Optionally specify axes, otherwise new plot will be created
    :param trange: Optionally specify the time range - useful when creating multiple plots
    :param colourbar: If True, a colourbar is added
    :return:
    """
    buff = 0.01

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    # get full domain boundary
    dom = ops.cascaded_union([boundaries[k] for k in ht])
    xmin, ymin, xmax, ymax = dom.bounds
    if trange is None:
        trange = (np.min(ht.values()), np.max(ht.values()))

    sc_map = colour_mapper([], vmin=trange[0], vmax=trange[1], cmap='Reds')
    for k in boundaries:
        if k in ht:
            plot_shapely_geos(boundaries[k], ec='k', fc=sc_map.to_rgba(ht[k]), ax=ax)
        else:
            plot_shapely_geos(boundaries[k], ec='k', fc='none', ax=ax)
        if k in hd:
            centroid = boundaries[k].centroid
            circ = Circle((centroid.x, centroid.y), radius=hd[k], edgecolor='none', facecolor='b')
            ax.add_patch(circ)


    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim([xmin - dx * buff, xmax + dx * buff])
    ax.set_ylim([ymin - dy * buff, ymax + dy * buff])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout(0.)
