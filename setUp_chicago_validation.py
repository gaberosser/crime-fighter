__author__ = 'gabriel'
import numpy as np
import datetime
from analysis import plotting, chicago
import dill
from point_process import estimation
from database.models import ChicagoDivision


# with open('assess_by_one_month2.pickle', 'w') as f:
#     callback = lambda data: dill.dump(data, f)
#     res2 = chicago.validate_point_process_multi(
#         start_date=datetime.datetime(2002, 1, 1),
#         training_size=30,
#         num_validation=1,
#         # num_validation=20,
#         num_pp_iter=20,
#         grid_size=100,
#         dt=100,
#         callback_func=callback
#     )


# with open('assess_by_three_months.pickle', 'w') as f:
#     callback = lambda data: pickle.dump(data, f)
#     res2 = chicago.validate_point_process_multi(
#         start_date=datetime.datetime(2002, 1, 1),
#         training_size=90,
#         num_validation=20,
#         num_pp_iter=20,
#         grid_size=500,
#         dt=90,
#         callback_func=callback
#     )
#
#
# with open('assess_by_year.pickle', 'w') as f:
#     callback = lambda data: pickle.dump(data, f)
#     res1 = chicago.validate_point_process_multi(
#         start_date=datetime.datetime(2002, 1, 1),
#         training_size=360,
#         num_validation=20,
#         num_pp_iter=20,
#         grid_size=500,
#         dt=180,
#         callback_func=callback
#     )

start_date = datetime.datetime(2011, 9, 27)
estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 500,
    'bg_kde_kwargs': {'number_nn': [101, 16]},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.3, 5, 5]},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs)
}

chic = chicago.compute_chicago_region()
southwest = ChicagoDivision.objects.get(name='Southwest').mpoly
south = ChicagoDivision.objects.get(name='South').mpoly
west = ChicagoDivision.objects.get(name='West').mpoly

southwest_buf = southwest.buffer(1500)
south_buf = south.buffer(1500)
west_buf = west.buffer(1500)

res, vb = chicago.validate_point_process(
    start_date=start_date,
    training_size=60,
    num_validation=5,
    domain=chic,
    model_kwargs=model_kwargs
)