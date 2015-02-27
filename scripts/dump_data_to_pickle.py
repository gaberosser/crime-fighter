__author__ = 'gabriel'
import pickle
import os
from analysis import cad, chicago
from database import models
import datetime
from analysis.spatial import geodjango_to_shapely

"""
This script loads CAD and Chicago data from the database and dumps it to a pickled array for use on Legion
"""

ROOT_DIR = '/home/gabriel/pickled_data/'
if not os.path.isdir(ROOT_DIR):
    os.makedirs(ROOT_DIR)

# regional boundaries

# chicago_south = models.ChicagoDivision.objects.get(name='South').mpoly.simplify()  # need this below

boundaries = {
    'camden': cad.get_camden_region(),
    'chicago': chicago.compute_chicago_region(),
    'chicago_south': models.ChicagoDivision.objects.get(name='South').mpoly,
}

for k in boundaries:
    boundaries[k] = geodjango_to_shapely(boundaries[k]).simplify(0)

with open(os.path.join(ROOT_DIR, 'boundaries.pickle'), 'w') as f:
    pickle.dump(boundaries, f)

## CAMDEN

start_date = datetime.datetime(2011, 3, 1)

# define crime types

crime_types = {
    'burglary': 3,
    'robbery': 5,
    'theft_of_vehicle': 6,
    'violence': 1,
}

for k, n in crime_types.items():
    this_path = os.path.join(ROOT_DIR, 'camden')
    if not os.path.isdir(this_path):
        os.makedirs(this_path)
    data, t0, cid = cad.get_crimes_by_type(n)
    with open(os.path.join(this_path, '%s.pickle' % k), 'w') as f:
        pickle.dump(data, f)

## CHICAGO

end_date = start_date + datetime.timedelta(days=277 + 480)

crime_types = {
    'burglary': 'burglary',
    'robbery': 'robbery',
    'theft_of_vehicle': 'motor vehicle theft',
    'violence': 'assault',
}

for k, n in crime_types.items():
    this_path = os.path.join(ROOT_DIR, 'chicago_south')
    if not os.path.isdir(this_path):
        os.makedirs(this_path)
    data, t0, cid = chicago.get_crimes_by_type(crime_type=n,
                                               start_date=start_date,
                                               end_date=end_date,
                                               domain=boundaries['chicago_south'])

    with open(os.path.join(this_path, '%s.pickle' % k), 'w') as f:
        pickle.dump(data, f)