from ..network_loader import ITNLoader
from ..data_loader import DataLoaderFile, DataLoaderDB, datetime_to_days
try:
    from database import models
    NO_DB = False
except ImportError:
    # disable loading from DB
    NO_DB = True
from database.consts import SRID
import consts
import datetime
import re
import numpy as np
from shapely import wkb, geometry
import fiona
import pickle


class CamdenNetworkLoader(ITNLoader):
    srid = 27700
    net_dir = consts.NETWORK_DIR


def load_network():
    obj = CamdenNetworkLoader('mastermap-itn_camden_buff2000.gml')
    return obj.load()


def load_boundary():
    with open(consts.BOUNDARY_FILE, 'r') as f:
        x, y = pickle.load(f)
    return geometry.Polygon(zip(x, y))

