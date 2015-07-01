__author__ = 'gabriel'
import math
import numpy as np
from shapely import geometry
import shapefile


try:
    from django.contrib.gis import geos
    HAS_GEODJANGO = True
except ImportError:
    geos = None
    HAS_GEODJANGO = False

def geodjango_to_shapely(geos_obj):
    """ Convert geodjango geometry to shapely for plotting etc
        inputs: x is a sequence of geodjango geometry objects """
    assert HAS_GEODJANGO, "Requires Geodjango"

    geodjango_poly_to_shapely = lambda t: geometry.Polygon(shell=t.coords[0], holes=t.coords[1:])

    converters = {
        geos.Point: lambda t: geometry.Point(t.coords),
        geos.LineString: lambda t: geometry.LineString(t.coords),
        geos.Polygon: lambda t: geodjango_poly_to_shapely(t),
        geos.MultiPolygon: lambda t: geometry.MultiPolygon([geodjango_poly_to_shapely(x) for x in t])
    }

    if not issubclass(geos_obj.__class__, geos.GEOSGeometry):
        raise TypeError("Require object that inherits from geos.GEOSGeometry")

    return converters[type(geos_obj)](geos_obj)   # FIXME: why is PyCharm complaining about this line?!


def bounding_box_grid(spatial_domain, grid_length, offset_coords=None):
    """
    Compute a grid on the bounding box of the supplied domain.
    :param spatial_domain: geos Polygon or Multipolygon describing the overall geometry
    :param grid_length: the length of one side of the grid square
    :param offset_coords: tuple giving the (x, y) coordinates of the bottom LHS of a gridsquare, default = (0, 0)
    :return: array of x an dy coords of vertices
    """
    offset_coords = offset_coords or (0, 0)

    if HAS_GEODJANGO and isinstance(spatial_domain, geos.Polygon):
        spatial_domain = geodjango_to_shapely(spatial_domain)

    xmin, ymin, xmax, ymax = spatial_domain.bounds

    # 1) create grid over entire bounding box
    sq_x_l = math.ceil((offset_coords[0] - xmin) / grid_length)
    sq_x_r = math.ceil((xmax - offset_coords[0]) / grid_length)
    sq_y_l = math.ceil((offset_coords[1] - ymin) / grid_length)
    sq_y_r = math.ceil((ymax - offset_coords[1]) / grid_length)
    edges_x = grid_length * np.arange(-sq_x_l, sq_x_r + 1) + offset_coords[0]
    edges_y = grid_length * np.arange(-sq_y_l, sq_y_r + 1) + offset_coords[1]

    return edges_x, edges_y


def create_spatial_grid(spatial_domain, grid_length, offset_coords=None):
    """
    Compute a grid on the spatial domain.
    :param spatial_domain: geos Polygon or Multipolygon describing the overall geometry
    :param grid_length: the length of one side of the grid square
    :param offset_coords: tuple giving the (x, y) coordinates of the bottom LHS of a gridsquare, default = (0, 0)
    :return: list of grid vertices and centroids (both (x,y) pairs)
    """

    if HAS_GEODJANGO and isinstance(spatial_domain, geos.Polygon):
        polygon = geos.Polygon
    elif isinstance(spatial_domain, geometry.Polygon):
        polygon = geometry.Polygon
    else:
        raise TypeError("spatial_domain must be of type django.contrib.gis.geos.Polygon or shapely.geometry.Polygon")

    intersect_polys = []
    full_extents = []
    full_grid_square = []
    edges_x, edges_y = bounding_box_grid(spatial_domain, grid_length, offset_coords=offset_coords)

    for ix in range(len(edges_x) - 1):
        for iy in range(len(edges_y) - 1):
            p = polygon((
                (edges_x[ix], edges_y[iy]),
                (edges_x[ix+1], edges_y[iy]),
                (edges_x[ix+1], edges_y[iy+1]),
                (edges_x[ix], edges_y[iy+1]),
                (edges_x[ix], edges_y[iy]),
            ))
            if p.within(spatial_domain):
                intersect_polys.append(p)
                full_extents.append(p.bounds)
                full_grid_square.append(True)
            elif spatial_domain.intersects(p):
                intersect_polys.append(spatial_domain.intersection(p))
                full_extents.append(p.bounds)
                full_grid_square.append(False)

    return intersect_polys, full_extents, full_grid_square


def shapely_rectangle_from_vertices(xmin, ymin, xmax, ymax):
    return geometry.Polygon([
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ])


def geodjango_rectangle_from_vertices(xmin, ymin, xmax, ymax):
    return geos.Polygon([
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ])

def write_polygons_to_shapefile(outfile, polygons, field_description=None, **other_attrs):
    """

    :param outfile:
    :param polygons: List of shapely polygons
    :param field_description: dictionary, each key is a field name, each value is a dict
    ('fieldType': 'C', 'size: '50')
    fieldType may be 'C' for char, 'N' for number.
    :param other_attrs: arrays of equal length to polygons, one for each field in field_description.
    """
    w = shapefile.Writer(shapefile.POLYGON)
    for fieldname, fieldvals in field_description.items():
        w.field(fieldname, **fieldvals)
    for i, p in enumerate(polygons):
        parts = [list(t) for t in zip(*p.boundary.xy)]
        w.poly(parts=[parts])
        w.record(*[other_attrs[k][i] for k in field_description])
    w.save(outfile)

def is_clockwise(poly):
    c = np.array(poly.exterior.coords)
    dx = np.diff(c[:, 0])
    ay = c[:-1, 1] + c[1:, 1]
    t = dx * ay
    t[ay == 0] = 0.
    return np.sum(t) > 0


def is_self_bounding(poly):
    # check whether this poly is equivalent to its bounding box, i.e. a rectangle
    a = poly.boundary
    b = poly.envelope.boundary
     ## TODO: finish or discard


def random_points_within_poly(poly, npts):
    """
    Generate n point coordinates that lie within poly
    NB this can be VERY SLOW if the polygon does not occupy much of its bounding box
    :return: x, y
    """
    try:
        # geodjango/OGR interface
        xmin, ymin, xmax, ymax = poly.extent
        is_geodjango = True
    except AttributeError:
        # shapely interface
        xmin, ymin, xmax, ymax = poly.bounds
        is_geodjango = False
    dx = xmax - xmin
    dy = ymax - ymin
    out_idx = np.ones(npts).astype(bool)
    x = np.zeros(npts)
    y = np.zeros(npts)

    while out_idx.sum():
        xn = np.random.random(size=out_idx.sum()) * dx + xmin
        yn = np.random.random(size=out_idx.sum()) * dy + ymin
        x[out_idx] = xn
        y[out_idx] = yn
        if is_geodjango:
            out_idx = np.array([not geos.Point(a, b).within(poly) for (a, b) in zip(x, y)])
        else:
            out_idx = np.array([not geometry.Point(a, b).within(poly) for (a, b) in zip(x, y)])

    return x, y


def jiggle_on_grid_points(data, grid_polys):
    """
    Introduce random jiggle to all points in the data whose spatial location lies exactly on the centroid of a grid
    polygon.
    :param data: Numpy array, as returned by get_crimes_by_type
    :param grid_polys: Iterable containing geos polygon objects
    :return: new_data - with same times as the original events, but jiggled where necessary
    """
    new_data = []
    centroids = np.array([t.centroid.coords for t in grid_polys])
    for t in data:
        x = t[1]
        y = t[2]
        idx = np.where(np.sum(centroids == t[1:], axis=1) == 2)[0]
        if idx:
            assert len(idx) == 1, "Overlapping polygons are not supported"
            idx = idx[0]
            this_datum = (t[0],) + random_points_within_poly(grid_polys[idx], 1)
        else:
            this_datum = t
        new_data.append(this_datum)

    return np.array(new_data)