__author__ = 'gabriel'
from data.models import CartesianSpaceTimeData
import seaborn as sns
import pandas
import numpy as np
from utils import mask_contour
from shapely import geometry as shapely_geometry
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib import collections
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from descartes import PolygonPatch
import json
from analysis.spatial import is_clockwise, bounding_box_grid, geodjango_to_shapely
from point_process.utils import linkages, linkage_func_separable


def pairwise_distance_histogram(data,
                                max_t,
                                max_d,
                                remove_coincident_pairs=False,
                                nbin=40,
                                vmax=None,
                                fmax=None):
    data = CartesianSpaceTimeData(data)
    linkage_fun = linkage_func_separable(max_t, max_d)
    i, j = linkages(data, linkage_fun,
                    remove_coincident_pairs=remove_coincident_pairs)
    interpoint = data[j] - data[i]
    df = pandas.DataFrame(interpoint[:, 1:], columns=('x (m)', 'y (m)'))

    if vmax is not None:
        joint_kws = dict(vmax=vmax)
    else:
        joint_kws = dict()

    sns.set_context("paper", font_scale=2.)
    with sns.axes_style("white"):
        grid = sns.jointplot(x='x (m)', y='y (m)', data=df, kind='hex', color='k',
                             stat_func=None, space=0,
                             marginal_kws=dict(bins=nbin),
                             joint_kws=joint_kws,
                             size=8)
        if fmax is not None:
            pc = [t for t in grid.ax_joint.get_children() if isinstance(t, collections.PolyCollection)][0]
            cdata = pc.get_array()
            plt.close(grid.fig)
            cdata.sort()
            vmax = cdata[int(len(cdata) * fmax)]

            grid = sns.jointplot(x='x (m)', y='y (m)', data=df, kind='hex', color='k',
                                 stat_func=None, space=0,
                                 marginal_kws=dict(bins=nbin),
                                 joint_kws=dict(vmax=vmax),
                                 size=8)

    return grid