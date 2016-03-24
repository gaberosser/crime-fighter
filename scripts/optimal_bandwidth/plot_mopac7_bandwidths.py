__author__ = 'gabriel'
from scripts import OUT_DIR
import os
import numpy as np
import dill
from scripts.optimal_bandwidth.plots import plot_optimal_bandwidth_map
from analysis.london import get_borough_polys
from matplotlib import pyplot as plt


BOROUGH_NAME_MAP = {
    'BS': 'Kensington and Chelsea',
    'CW': 'Westminster',
    'EK': 'Camden',
    'FH': 'Hammersmith and Fulham',
    'GD': 'Hackney',
    'HT': 'Tower Hamlets',
    'JC': 'Waltham Forest',
    'JI': 'Redbridge',
    'KD': 'Havering',
    'KF': 'Newham',
    'KG': 'Barking and Dagenham',
    'LX': 'Lambeth',
    'MD': 'Southwark',
    'NI': 'Islington',
    'PL': 'Lewisham',
    'PY': 'Bromley',
    'QA': 'Harrow',
    'QK': 'Brent',
    'RG': 'Greenwich',
    'RY': 'Bexley',
    'SX': 'Barnet',
    'TW': 'Richmond upon Thames',
    'TX': 'Hounslow',
    'VK': 'Kingston upon Thames',
    'VW': 'Merton',
    'WW': 'Wandsworth',
    'XB': 'Ealing',
    'XH': 'Hillingdon',
    'YE': 'Enfield',
    'YR': 'Haringey',
    'ZD': 'Croydon',
    'ZT': 'Sutton',
    'XXX': 'City of London',
}
names_by_code = dict([(v, k.lower()) for k, v in BOROUGH_NAME_MAP.items()])

boundaries = get_borough_polys()
boundaries_by_code = dict([(names_by_code[k], v) for k, v in boundaries.items()])
indir = os.path.join(OUT_DIR, 'greater_london_aggregated_likelihoods')
crime_types = (
    'burglary_in_a_dwelling',
    'criminal_damage',
    'robbery',
    'theft_person',
    'theft_from_motor_vehicle',
    'theft_or_taking_of_motor_vehicle',
)

lls = {}
ht = {}
hd = {}

for ct in crime_types:
    fn = os.path.join(indir, '%s.pkl' % ct)
    hd[ct] = {}
    ht[ct] = {}
    with open(fn, 'rb') as f:
        tt, dd, lls[ct] = dill.load(f)
        for bo in lls[ct]:
            idx = np.argmax(lls[ct][bo])
            i, j = np.unravel_index(idx, tt.shape)
            ht[ct][bo] = tt[i, j]
            hd[ct][bo] = dd[i, j]
    plot_optimal_bandwidth_map(ht[ct], hd[ct], boundaries_by_code, trange=[30, 120.])
    plt.title(ct.replace('_', ' ').capitalize())
    plt.gcf().savefig('%s.png' % ct, dpi=200)
