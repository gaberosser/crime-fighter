__author__ = 'gabriel'
import numpy as np
from analysis import cad, chicago
import datetime
from database import models

# end_date is the LAST DAY OF TRAINING
# NB make it a date and any crimes up to the very end of that date are included
end_date = datetime.date(2011, 12, 2)
# equivalent in number of days from t0 (1/3/2011)
end_day_number = 276

## Camden
print "Camden"

poly = cad.get_camden_region()
qset = models.Division.objects.filter(type='cad_250m_grid')
qset = sorted(qset, key=lambda x:int(x.name))
grid_squares = [t.mpoly[0] for t in qset]
centroids = np.array([t.centroid.coords for t in grid_squares])

nicl_types = {
    'burglary': 3,
    'robbery': 5,
    'theft of vehicle': 6,
    'violence': 1,
}

for name, n in nicl_types.items():

    data, t0, cid = cad.get_crimes_by_type(nicl_type=n, end_date=end_date)
    snapped_idx = [i for i, t in enumerate(data) if np.any(centroids == t[1:])]
    print name
    print "Total crimes: %d. Snapped to grid: %d" % (len(data), len(snapped_idx))


## Chicago South side
print "Chicago South side"
start_date = datetime.date(2011, 3, 1)
south = models.ChicagoDivision.objects.get(name='South').mpoly


crime_types = [
    'burglary',
    'robbery',
    'motor vehicle theft',
    'assault'
]

for name in crime_types:
    data, t0, cid = chicago.get_crimes_by_type(name, start_date=start_date,
                                               end_date=end_date,
                                               domain=south)
    print name
    print "Total crimes: %d" % len(data)