from settings import DATA_DIR
import os
from network import itn
from jdi.data import consts
import datetime


def get_itn_network(borough=None):
    if borough is None:
        infile = os.path.join(DATA_DIR, 'greater_london', 'greater_london_network.pickle')
    else:
        #t = consts.BOROUGH_NAME_MAP[borough.upper()]
        infile = os.path.join(DATA_DIR, 'greater_london', 'itn_net_by_borough_buffer500', '%s.net' % borough.lower())
    return itn.ITNStreetNet.from_pickle(infile)
    
        
if __name__ == "__main__":
    """
    Requested by Huanfa early 12/2015.
    Create shapefile with network segments and crime counts in a defined time
    window.    
    """
    from jdi.data import cris
    
    borough = 'ek'
    crime_types = (
        'THEFT',
        'BURGLARY',
        'HOMICIDE',
        'BATTERY',
        'ARSON',
        'MOTOR VEHICLE THEFT',
        'ASSAULT',
        'ROBBERY',
    )    
    crime_type = ('burglary', )
    start_dt = datetime.datetime(2014, 1, 1)
    end_dt = datetime.datetime(2014, 7, 1)
    borough_full = consts.BOROUGH_NAME_MAP[borough.upper()]
    # get data
    data, t0, cid = cris.get_cris_data(borough=borough, start_dt=start_dt, end_dt=end_dt)
    pass