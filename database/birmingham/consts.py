__author__ = 'gabriel'
import os
from settings import DATA_DIR


DATA_SUBDIR = os.path.join(DATA_DIR, 'birmingham')
NETWORK_DIR = os.path.join(DATA_SUBDIR, 'network')
CRIME_DATA_FILE = os.path.join(DATA_SUBDIR, 'data_090301_140831_matched.csv')