from settings import DATA_DIR
import fiona
from shapely import geometry
import os


def get_borough_boundary(name=None):
    shp = os.path.join(DATA_DIR, 
    'greater_london', 
    'boundaries', 
    'borough',
    'boroughs_with_mps_code.shp')
    
    if name is None:
        res = {}
    
    with fiona.open(shp) as s:
        for i, t in s.items():
            if name is None:
                if t['properties']['MPS_CODE'] is None:
                    continue            
                res[t['properties']['MPS_CODE'].lower()] = geometry.shape(t['geometry'])
            elif t['properties']['MPS_CODE'].lower() == name.lower():
                return geometry.shape(t['geometry'])
    
    if name is None:
        return res