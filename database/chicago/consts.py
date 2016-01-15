__author__ = 'gabriel'
import os
from settings import DATA_DIR


DATA_SUBDIR = os.path.join(DATA_DIR, 'chicago')

REGIONS = (
    'Northwest',    'Far North',        'North',
    'West',         'Central',          'South',
    'Southwest',    'Far Southwest',    'Far Southeast',
)

CRIME_TYPES = (
    'burglary',
    'assault',
)

FILE_FRIENDLY_REGIONS = {
    'South': 'chicago_south',
    'Southwest': 'chicago_southwest',
    'West' : 'chicago_west',
    'Northwest': 'chicago_northwest',
    'North': 'chicago_north',
    'Central': 'chicago_central',
    'Far North': 'chicago_far_north',
    'Far Southwest': 'chicago_far_southwest',
    'Far Southeast': 'chicago_far_southeast',
}

ABBREVIATED_REGIONS = {
    'South': 'S',
    'Southwest': 'SW',
    'West': 'W',
    'Northwest': 'NW',
    'North': 'N',
    'Central': 'C',
    'Far North': 'FN',
    'Far Southwest': 'FSW',
    'Far Southeast': 'FSE',
}


NETWORK_DIR = os.path.join(DATA_SUBDIR, 'network')
