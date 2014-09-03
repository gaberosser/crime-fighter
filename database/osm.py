__author__ = 'gabriel'
from django.contrib.gis import geos
from django.db import connections, connection
from matplotlib import pyplot as plt
from analysis import plotting
import collections
import models

SRID = 2028
DBNAME = connections.databases['default']['NAME']
cursor = connection.cursor()

def get_points_within(poly, faster=False):
    # TODO: faster specifies simple bounding box lookup rather than proper intersection.
    # Implement or remove as necessary
    qry = '''SELECT ST_AsText(ST_Transform(way, {0})) FROM planet_osm_point
             WHERE ST_Intersects(ST_Transform(way, {0}), ST_GeomFromText('{1}', {0}))'''.format(SRID, poly.wkt)

    cursor.execute(qry)
    return cursor.fetchall()


def get_lines_within(poly, faster=False):
    # TODO: faster specifies simple bounding box lookup rather than proper intersection.
    # Implement or remove as necessary
    qry = '''SELECT ST_AsText(ST_Transform(way, {0})) FROM planet_osm_line
             WHERE ST_Intersects(ST_Transform(way, {0}), ST_GeomFromText('{1}', {0}))'''.format(SRID, poly.wkt)

    cursor.execute(qry)
    return cursor.fetchall()


def get_roads_within(poly, types=None, faster=False):
    # TODO: faster specifies simple bounding box lookup rather than proper intersection.
    # Implement or remove as necessary

    if types is None:
        # default list of road types
        types = [
            'motorway',
            'motorway link',
            'primary',
            'primary link',
            'secondary',
            'secondary link',
            'residential',
            'trunk'
        ]
    types = tuple(types)
    if len(types) == 1:
        types = "('" + types[0] + "')"
    else:
        types = str(types)

    qry = '''SELECT highway, ST_AsText(ST_Transform(way, {0})) FROM planet_osm_roads
             WHERE ST_Intersects(ST_Transform(way, {0}), ST_GeomFromText('{1}', {0}))
             AND LOWER(highway) IN {2};'''.format(SRID, poly.wkt, types)

    cursor.execute(qry)
    return cursor.fetchall()


class Osm(object):

    line_cats = {
        'highway': [
            'motorway',
            'motorway link',
            'primary',
            'primary link',
            'secondary',
            'secondary link',
            'residential',
            'trunk',
        ],
        'railway': [
            'rail',
        ],
        'waterway': [
            'river'
        ],
    }

    poly_cats = {
        'natural': [
            'water',
            'lake'
        ],
        'water': [
            'pond',
            'river',
            'lake',
            'reservoir'
        ],
        'aeroway': [
            'terminal',
        ]
    }

    def __init__(self, domain, srid=None, buffer=1500):
        self.cursor = connection.cursor()
        self.domain = domain
        if not srid:
            srid = domain.srid
        self.srid = srid
        self.buffer = buffer
        self.elements = None
        self.update_all()

    @property
    def all_keys(self):
        return self.line_cats.keys() + self.poly_cats.keys()

    def reset_elements(self):
        self.elements = dict([(k, collections.defaultdict(list)) for k in self.all_keys])

    def update_all(self):
        buffered_domain = self.domain.buffer(self.buffer)
        self.reset_elements()
        # paths
        for k in self.line_cats.keys():
            for x in self._getter('line', k, buffered_domain):
                self.elements[k][x[0]].append(x[-1])
        # polys
        for k in self.poly_cats.keys():
            for x in self._getter('poly', k, buffered_domain):
                self.elements[k][x[0]].append(x[-1])

    def _getter(self, category, type_, domain):
        if category == 'line':
            cats = self.line_cats[type_]
            table = 'planet_osm_roads'
        elif category == 'poly':
            cats = self.poly_cats[type_]
            table = 'planet_osm_polygon'
        else:
            raise AttributeError("Category %s not recognised", category)

        cats = tuple(cats)
        if len(cats) == 1:
            cats = "('" + cats[0] + "')"
        else:
            cats = str(cats)

        qry = '''SELECT "{1}", ST_AsText(ST_Transform(way, {2})) FROM {0}
             WHERE ST_Intersects(ST_Transform(way, {2}), ST_GeomFromText('{3}', {2}))
             AND LOWER("{1}") IN {4};'''.format(table, type_, self.srid, domain, cats)

        self.cursor.execute(qry)
        return [x[:-1] + (geos.fromstr(x[-1]),) for x in self.cursor.fetchall()]


    def path_getter(self, type_, domain, cats=None):
        if cats is None:
            cats = self.line_cats[type_]
        cats = tuple(cats)
        if len(cats) == 1:
            cats = "('" + cats[0] + "')"
        else:
            cats = str(cats)

        qry = '''SELECT {0}, ST_AsText(ST_Transform(way, {1})) FROM planet_osm_roads
             WHERE ST_Intersects(ST_Transform(way, {1}), ST_GeomFromText('{2}', {1}))
             AND LOWER({0}) IN {3};'''.format(type_, self.srid, domain, cats)

        self.cursor.execute(qry)
        return [x[:-1] + (geos.fromstr(x[-1]),) for x in self.cursor.fetchall()]


class OsmRendererBase(object):

    style = {
        'domain': {'linewidth': 2, 'edgecolor': 'k', 'facecolor': (0, 0, 0, 0.1)},
        'highway': {
            'motorway': {'linewidth': 3, 'color': 'orange', 'alpha': 0.5},
            'primary': {'linewidth': 2, 'color': (1, 1, 0, 0.5)},
        },
        'natural': {
            'water': {'edgecolor': '#1C73FF', 'linewidth': 1, 'facecolor': '#1CCAFF'},
            'lake': {'edgecolor': '#1C73FF', 'linewidth': 1, 'facecolor': '#1CCAFF'},
        },
        'water': {
            'pond': {'edgecolor': '#1C73FF', 'linewidth': 1, 'facecolor': '#1CCAFF'},
            'river': {'edgecolor': '#1C73FF', 'linewidth': 1, 'facecolor': '#1CCAFF'},
            'lake': {'edgecolor': '#1C73FF', 'linewidth': 1, 'facecolor': '#1CCAFF'},
            'reservoir': {'edgecolor': '#1C73FF', 'linewidth': 1, 'facecolor': '#1CCAFF'},
        },
        'aeroway': {
            'terminal': {'linecolor': 'b', 'alpha': 0.3, 'edgecolor': 'b', 'linewidth': 0.5},
        }
        # 'railway': {
        #     'rail': {'linewidth': 1, 'color': (0, 0, 1, 0.6)},
        # }
    }

    def __init__(self, domain):
        self.osm = Osm(domain)

    def render(self, ax=None):
        ax = ax or plt.gca()
        plotting.plot_geodjango_shapes(self.osm.domain, ax=ax, set_axes=True, **self.style['domain'])
        for t, v in self.osm.elements.iteritems():
            if t in self.style:
                this_style = self.style[t]
                for k, x in v.items():
                    if k in this_style:
                        s = this_style[k]
                        plotting.plot_geodjango_shapes(x, ax=ax, set_axes=False, **s)
        ax.set_aspect('equal')
        return ax
