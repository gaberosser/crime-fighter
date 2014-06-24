__author__ = 'gabriel'
import math
from django.contrib.gis import geos
import numpy as np

def create_spatial_grid(spatial_domain, grid_length, offset_coords=None):
    """
    Compute a grid on the spatial domain.
    :param spatial_domain: geos Polygon or Multipolygon describing the overall geometry
    :param grid_length: the length of one side of the grid square
    :param offset_coords: tuple giving the (x, y) coordinates of the bottom LHS of a gridsquare, default = (0, 0)
    :return: list of grid vertices and centroids (both (x,y) pairs)
    """
    offset_coords = offset_coords or (0, 0)
    xmin, ymin, xmax, ymax = spatial_domain.extent

    # 1) create grid over entire bounding box
    sq_x_l = math.ceil((offset_coords[0] - xmin) / grid_length)
    sq_x_r = math.ceil((xmax - offset_coords[0]) / grid_length)
    sq_y_l = math.ceil((offset_coords[1] - ymin) / grid_length)
    sq_y_r = math.ceil((ymax - offset_coords[1]) / grid_length)
    edges_x = grid_length * np.arange(-sq_x_l, sq_x_r + 1) + offset_coords[0]
    edges_y = grid_length * np.arange(-sq_y_l, sq_y_r + 1) + offset_coords[1]
    polys = []
    for ix in range(len(edges_x) - 1):
        for iy in range(len(edges_y) - 1):
            p = geos.Polygon((
                (edges_x[ix], edges_y[iy]),
                (edges_x[ix+1], edges_y[iy]),
                (edges_x[ix+1], edges_y[iy+1]),
                (edges_x[ix], edges_y[iy+1]),
                (edges_x[ix], edges_y[iy]),
            ))
            if spatial_domain.intersects(p):
                polys.append(spatial_domain.intersection(p))

    return polys