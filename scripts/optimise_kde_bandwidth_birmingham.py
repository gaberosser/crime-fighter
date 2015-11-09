__author__ = 'gabriel'
import numpy as np
from analysis import birmingham
from kde import optimisation
import datetime
import settings
import os

START_DATE = datetime.date(2012, 7, 1)
n_train = 365
n_test = 10
END_DATE = START_DATE + datetime.timedelta(days=n_train + n_test - 1)


if __name__ == "__main__":
    # get data
    domain = birmingham.get_boundary()
    data, t0, cid = birmingham.get_crimes(
        start_date=START_DATE,
        end_date=END_DATE
    )
    ss, tt, res = optimisation.compute_log_likelihood_surface_fixed_bandwidth(
        data,
        n_train,
        n_test
    )
