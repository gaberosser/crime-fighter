__author__ = 'gabriel'
from database import models
from point_process import validate, models as pp_models, simulate, plotting, estimation, plotting
import numpy as np
import datetime
import settings
import os
import dill
from analysis import chicago, cad

start_date = datetime.datetime(2011, 3, 1)
end_date = start_date + datetime.timedelta(days=277 + 480)
cutoff_day_number = 277
niter = 75

estimate_kwargs = {
    'ct': 1,
    'cd': 0.02,
    'frac_bg': 0.5,
}

model_kwargs = {
    'parallel': True,
    'max_delta_t': 120, # set on each iteration
    'max_delta_d': 500, # set on each iteration
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': None,
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers_fixed_proportion_bg(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
}


## CHICAGO SOUTH SIDE

# chic_south = models.ChicagoDivision.objects.get(name='South').mpoly

## Load from database
# res, t0, cid = chicago.get_crimes_by_type(crime_type='burglary', start_date=start_date, end_date=end_date,
#                                           domain=chic_south)

## Load from file
with open(os.path.join(settings.IN_DIR, 'chicago_south', 'burglary.pickle'), 'r') as f:
    res = dill.load(f)

## CAMDEN

## Load from database
# res, t0, cid = cad.get_crimes_by_type(nicl_type=3)  # burglary

training = res[res[:, 0] <= cutoff_day_number]

sepp_isotropic = pp_models.SeppStochasticNnIsotropicTrigger(data=training, **model_kwargs)
ps_isotropic = sepp_isotropic.train(niter=niter)

# sepp_xy = pp_models.SeppStochasticNn(data=training, **model_kwargs)
# ps_xy = sepp_xy.train(niter=niter)
