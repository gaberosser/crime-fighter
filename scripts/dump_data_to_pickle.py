__author__ = 'gabriel'
import pickle
import os
from analysis import cad, chicago, spatial
from database import models
import datetime
from analysis.spatial import geodjango_to_shapely
from . import IN_DIR as DATA_DIR

"""
This script loads CAD and Chicago data from the database and dumps it to a pickled array for use on Legion
"""

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

# regional boundaries

# chicago_south = models.ChicagoDivision.objects.get(name='South').mpoly.simplify()  # need this below

boundaries = {
    'camden': cad.get_camden_region(as_shapely=True).simplify(0),
    'chicago': chicago.compute_chicago_region(as_shapely=True).simplify(0),
}

chicago_sides = chicago.get_chicago_side_polys(as_shapely=True)
for k, v in chicago_sides.iteritems():
    key = 'chicago_' + k.lower().replace(' ', '_')
    boundaries[key] = v

with open(os.path.join(DATA_DIR, 'boundaries.pickle'), 'w') as f:
    pickle.dump(boundaries, f)

## CAMDEN

start_date = datetime.datetime(2011, 3, 1)
# snapping polys
grid_polys = [t.mpoly.simplify() for t in models.Division.objects.filter(type='cad_250m_grid')]

# define crime types

crime_types = {
    'burglary': 3,
    'robbery': 5,
    'theft_of_vehicle': 6,
    'violence': 1,
}

for k, n in crime_types.items():
    this_path = os.path.join(DATA_DIR, 'camden')
    if not os.path.isdir(this_path):
        os.makedirs(this_path)
    data, t0, cid = cad.get_crimes_by_type(n)
    with open(os.path.join(this_path, '%s.pickle' % k), 'w') as f:
        pickle.dump(data, f)
    # jiggle on-grid crimes
    jdata = spatial.jiggle_on_grid_points(data, grid_polys)
    with open(os.path.join(this_path, '%s_jiggle.pickle' % k), 'w') as f:
        pickle.dump(jdata, f)

## CHICAGO SIDES
crime_types = {
    'burglary': 'burglary',
    'robbery': 'robbery',
    'motor_vehicle_theft': 'motor vehicle theft',
    'assault': 'assault',
}

for k in chicago_sides.keys():
    key = 'chicago_' + k.lower().replace(' ', '_')
    domain = chicago_sides[k]

    for ct, n in crime_types.items():
        this_path = os.path.join(DATA_DIR, 'chicago', key)
        if not os.path.isdir(this_path):
            os.makedirs(this_path)
        data, t0, cid = chicago.get_crimes_by_type(crime_type=n,
                                                   start_date=None,
                                                   end_date=None,
                                                   domain=domain)

        with open(os.path.join(this_path, '%s.pickle' % ct), 'w') as f:
            pickle.dump((data, t0, cid), f)

## CHICAGO SOUTH

end_date = start_date + datetime.timedelta(days=277 + 480)

crime_types = {
    'burglary': 'burglary',
    'robbery': 'robbery',
    'theft_of_vehicle': 'motor vehicle theft',
    'violence': 'assault',
}

for k, n in crime_types.items():
    this_path = os.path.join(DATA_DIR, 'chicago_south')
    if not os.path.isdir(this_path):
        os.makedirs(this_path)
    data, t0, cid = chicago.get_crimes_by_type(crime_type=n,
                                               start_date=start_date,
                                               end_date=end_date,
                                               domain=boundaries['chicago_south'])

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
    this_path = os.path.join(DATA_DIR, 'chicago')
    if not os.path.isdir(this_path):
        os.makedirs(this_path)
    data, t0, cid = chicago.get_crimes_by_type(crime_type=n,
                                               start_date=start_date,
                                               end_date=end_date,
                                               domain=boundaries['chicago'])

    with open(os.path.join(this_path, '%s.pickle' % k), 'w') as f:
        pickle.dump(data, f)