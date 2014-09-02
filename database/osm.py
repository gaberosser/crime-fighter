__author__ = 'gabriel'
from django.contrib.gis import geos
from django.db import connections, connection

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