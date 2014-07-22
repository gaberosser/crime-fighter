__author__ = 'gabriel'
from point_process import validate, models, simulate, plotting, estimation, plotting
import numpy as np
import datetime
from analysis import plotting, chicago
from scipy import sparse
import pickle

res1 = chicago.validate_point_process_multi(
    start_date=datetime.datetime(2002, 1, 1),
    training_size=360,
    num_validation=20,
    num_pp_iter=20,
    grid_size=500,
    dt=180
)

with open('assess_by_year.pickle', 'w') as f:
    pickle.dump(res1, f)

res2 = chicago.validate_point_process_multi(
    start_date=datetime.datetime(2002, 1, 1),
    training_size=90,
    num_validation=20,
    num_pp_iter=20,
    grid_size=500,
    dt=90
)

with open('assess_by_three_months.pickle', 'w') as f:
    pickle.dump(res2, f)

