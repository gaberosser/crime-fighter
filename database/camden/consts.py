import os
from settings import DATA_DIR


DATA_SUBDIR = os.path.join(DATA_DIR, 'camden')
NETWORK_DIR = os.path.join(DATA_SUBDIR, 'network')
CRIME_DATA_FILE = os.path.join(DATA_SUBDIR, 'mar2011-mar2012.csv')
BOUNDARY_FILE = os.path.join(DATA_SUBDIR, 'boundary', 'boundary_line.pickle')