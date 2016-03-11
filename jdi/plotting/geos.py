import numpy as np
from shapely import geometry, ops
from matplotlib import pyplot as plt


def shapely_poly_to_xy(poly):
    xy = np.array(poly.exterior.coords)
    return xy[:,0], xy[:,1]

def shapely_mpoly_to_xy(mpoly):
    res = []
    for p in mpoly:
        res.append(shapely_poly_to_xy(p))


def plot_shapely_geos(res, ax=None, **plot_kwargs):
    ax = ax or plt.gca()
    if not hasattr(res, '__iter__'):
        res = [res]
    for r in res:
        if isinstance(r, geometry.Polygon):
            x, y = shapely_poly_to_xy(r)
            plt.plot(x, y, **plot_kwargs)
        elif isinstance(r, geometry.MultiPolygon):
            for p in r:
                plt.plot(*shapely_poly_to_xy(p), **plot_kwargs)
        