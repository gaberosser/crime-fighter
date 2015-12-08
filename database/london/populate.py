# from ..models import ArealUnit
from database.models import ArealUnit
from database.populate import sql_quote
import logging
import os
from settings import DATA_DIR
import shapefile
import fiona
from shapely import geometry
logger = logging.getLogger(__name__)


def populate_boroughs(verbose=True):

    infile = os.path.join(DATA_DIR, 'greater_london', 'boundaries', 'borough', 'London_Borough_Excluding_MHW.shp')
    data = []
    with fiona.open(infile) as src:
        for r in src:
            poly = geometry.shape(r['geometry'])
            datum = {
                'name': sql_quote(r['properties']['NAME']),
                'type': sql_quote('london_borough'),
                'country': sql_quote('en'),
                'planar_srid': 27700,
                'mpoly': "ST_Multi(ST_Transform(ST_GeomFromText('%s', 27700), 4326))" % poly.wkt,
            }
            data.append(datum)

    obj = ArealUnit()
    obj.insert_many(data)
    if verbose:
        logger.info("Added %d boroughs", len(data))


