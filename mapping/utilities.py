__author__ = 'gabriel'
import shapefile
from django.contrib.gis import geos
from database import models, logic
import os
import numpy as np
from settings import DATA_DIR


def get_camden_parks():
    parks_file = os.path.join(DATA_DIR, 'osm_london', 'natural')
    camden = models.Division.objects.get(type='borough', name='Camden').mpoly
    sf = shapefile.Reader(parks_file)
    recs = sf.records()
    shapes = sf.shapes()
    # get all parks
    park_idx = [i for i, x in enumerate(recs) if x[-1] == 'park']
    # park_shapes = [shapes[i] for i in park_idx]
    # park_recs = [recs[i] for i in park_idx]

    # restrict to camden
    camden_parks = {}
    for i in park_idx:
        ps = shapes[i]
        pr = recs[i]
        poly = geos.Polygon(np.array(ps.points), srid=4326)
        poly.transform(27700)
        if camden.intersects(poly):
            print "Intersection"
            name = pr[1]
            if name.strip(' '):
                camden_parks[name] = poly

    return camden_parks

