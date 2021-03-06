__author__ = 'gabriel'
# from django.contrib.gis import geos
from shapely import wkb, geometry
from django.db import connections, connection
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from plotting import spatial
import collections
import models

SRID = 2028
DBNAME = connections.databases['default']['NAME']
cursor = connection.cursor()

def sql_list(x):
    if hasattr(x, '__iter__'):
        if len(x) == 1:
            return "({0})".format(x[0])
        else:
            return str(tuple(x))
    else:
        return "({0})".format(x)


def sql_get_within(table, field, srid, domain, cats=None):
    a = "ST_Transform(way, {0})".format(srid)
    b = "ST_GeomFromText('{0}', {1})".format(domain.wkt, srid)
    qry = '''
          SELECT "{0}", {1} FROM {2}
          WHERE {1} && {3}
          AND ST_Intersects({1}, {3})
          '''.format(
        field,
        a,
        table,
        b
    )
    if cats is not None and cats != "*":
        qry += "AND LOWER({0}) IN {1}".format(field, sql_list(cats))
    qry += ";"
    return qry

#
# def get_points_within(poly):
#     qry = '''SELECT ST_AsText(ST_Transform(way, {0})) FROM planet_osm_point
#              WHERE ST_Transform(way, {0}) && ST_GeomFromText('{1}', {0})
#              AND ST_Intersects(ST_Transform(way, {0}), ST_GeomFromText('{1}', {0}))'''.format(SRID, poly.wkt)
#
#     cursor.execute(qry)
#     return cursor.fetchall()
#
#
# def get_lines_within(poly):
#     qry = '''SELECT ST_AsText(ST_Transform(way, {0})) FROM planet_osm_line
#              WHERE ST_Transform(way, {0}) && ST_GeomFromText('{1}', {0})
#              AND ST_Intersects(ST_Transform(way, {0}), ST_GeomFromText('{1}', {0}))'''.format(SRID, poly.wkt)
#
#     cursor.execute(qry)
#     return cursor.fetchall()
#
#
# def get_roads_within(poly, types=None):
#
#     if types is None:
#         # default list of road types
#         types = [
#             'motorway',
#             'motorway link',
#             'primary',
#             'primary link',
#             'secondary',
#             'secondary link',
#             'residential',
#             'trunk'
#         ]
#     types = tuple(types)
#     if len(types) == 1:
#         types = "('" + types[0] + "')"
#     else:
#         types = str(types)
#
#     qry = '''SELECT highway, ST_AsText(ST_Transform(way, {0})) FROM planet_osm_roads
#              WHERE ST_Transform(way, {0}) && ST_GeomFromText('{1}', {0})
#              AND ST_Intersects(ST_Transform(way, {0}), ST_GeomFromText('{1}', {0}))
#              AND LOWER(highway) IN {2};'''.format(SRID, poly.wkt, types)
#
#     cursor.execute(qry)
#     return cursor.fetchall()


class Osm(object):

    line_cats = {
        'highway': [
            'motorway',
            'motorway link',
            'primary',
            'primary link',
            'secondary',
            'secondary link',
            'tertiary',
            'tertiary_link',
            'residential',
            'trunk',
            'pedestrian',
            'unclassified',
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
        ],
        'leisure': '__all',
        'landuse': [
            'forest',
        ],
        'railway': [
            'station',
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
            # for x in self._getter('line', k, buffered_domain):
            for x in self.road_getter(k, buffered_domain):
                self.elements[k][x[0]].append(x[-1])

        # polys
        for k in self.poly_cats.keys():
            # for x in self._getter('poly', k, buffered_domain):
            for x in self.poly_getter(k, buffered_domain):
                self.elements[k][x[0]].append(x[-1])

    # def _getter(self, category, type_, domain):
    #     if category == 'line':
    #         cats = self.line_cats[type_]
    #         table = 'planet_osm_line'
    #         func = get_lines_within
    #     elif category == 'poly':
    #         cats = self.poly_cats[type_]
    #         table = 'planet_osm_polygon'
    #         func = get_roads_within
    #     else:
    #         raise AttributeError("Category %s not recognised", category)
    #
    #     # put categories in correct format for query
    #     if cats == '__all':
    #         b_cats = False
    #     else:
    #         b_cats = True
    #         cats = tuple(cats)
    #         if len(cats) == 1:
    #             cats = "('" + cats[0] + "')"
    #         else:
    #             cats = str(cats)
    #
    #     qry = '''SELECT "{1}", ST_AsText(ST_Transform(way, {2})) FROM {0}
    #          WHERE ST_Intersects(ST_Transform(way, {2}), ST_GeomFromText('{3}', {2}))'''.format(
    #         table, type_, self.srid, domain)
    #
    #     if b_cats:
    #         qry += '''AND LOWER("{0}") IN {1};'''.format(type_, cats)
    #
    #     qry += ';'
    #
    #     self.cursor.execute(qry)
    #     return [x[:-1] + (geos.fromstr(x[-1]),) for x in self.cursor.fetchall()]

    def road_getter(self, type_, domain, cats=None):
        qry = sql_get_within(
            'planet_osm_roads',
            type_,
            self.srid,
            domain,
            cats=cats)
        self.cursor.execute(qry)
        return [x[:-1] + (wkb.loads(x[-1], hex=True),) for x in self.cursor.fetchall()]

    def poly_getter(self, type_, domain, cats=None):
        qry = sql_get_within(
            'planet_osm_polygon',
            type_,
            self.srid,
            domain,
            cats=cats)
        try:
            self.cursor.execute(qry)
        except Exception as exc:
            import ipdb; ipdb.set_trace()

        return [x[:-1] + (wkb.loads(x[-1], hex=True),) for x in self.cursor.fetchall()]



class OsmRendererBase(object):

    style = {
        'domain': {'linewidth': 2.5, 'edgecolor': 'k', 'facecolor': 'none', 'zorder': 3},
        'highway': {
            '__all__': {},
            'primary': {'buffer': 5, 'facecolor': '#FFC61C', 'edgecolor': 'none'},
            'trunk': {'buffer': 5, 'facecolor': '#FFC61C', 'edgecolor': 'none'},
            'primary_link': {'buffer': 5, 'facecolor': '#FFC61C', 'edgecolor': 'none'},
            'secondary': {'buffer': 3.5, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'secondary_link': {'buffer': 3.5, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'tertiary': {'buffer': 2.5, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'tertiary_link': {'buffer': 2.5, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'residential': {'buffer': 2, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'pedestrian': {'buffer': 2, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'service': {'buffer': 2, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
            'unclassified': {'buffer': 2, 'facecolor': 'gray', 'alpha': 0.7, 'edgecolor': 'none'},
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
            'terminal': {'alpha': 0.3, 'edgecolor': 'gray', 'linewidth': 0.5},
        },
        'leisure': {
            'park': {'facecolor': '#5BE35B', 'alpha': 0.4, 'edgecolor': 'none'},
        },
        'landuse': {
            'forest': {'facecolor': '#5BE35B', 'alpha': 0.4, 'edgecolor': 'none'},
        },
        'railway': {
            'station': {'edgecolor': 'gray', 'linewidth': 1, 'facecolor': '#B2B2B2'},
        }
        # 'railway': {
        #     'rail': {'linewidth': 1, 'color': (0, 0, 1, 0.6)},
        # }
    }

    def __init__(self, domain, **kwargs):
        self.osm = Osm(domain, **kwargs)

    def render(self, ax=None):
        ax = ax or plt.gca()
        spatial.plot_shapely_geos(self.osm.domain, ax=ax, set_axes=True, **self.style['domain'])
        for t, v in self.osm.elements.iteritems():
            if t in self.style:
                this_style = self.style[t]
                for k, x in v.items():
                    if k in this_style:
                        s = dict(this_style[k])
                        buffer = s.pop('buffer', None)
                        if buffer:
                            # buffer each entry in list of linestrings
                            x = [a.buffer(buffer) for a in x]
                        spatial.plot_shapely_geos(x, ax=ax, set_axes=False, **s)
                    elif '__other' in this_style:
                        s = this_style['__other']
                        spatial.plot_shapely_geos(x, ax=ax, set_axes=False, **s)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
