__author__ = 'gabriel'
import numpy as np
import datetime
from analysis import plotting, chicago
import dill


with open('assess_by_one_month2.pickle', 'w') as f:
    callback = lambda data: dill.dump(data, f)
    res2 = chicago.validate_point_process_multi(
        start_date=datetime.datetime(2002, 1, 1),
        training_size=30,
        num_validation=1,
        # num_validation=20,
        num_pp_iter=20,
        grid_size=100,
        dt=100,
        callback_func=callback
    )


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