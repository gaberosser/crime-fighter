__author__ = 'gabriel'
from database import models
from point_process import validate, models as pp_models, simulate, plotting, estimation, plotting
import numpy as np
import datetime
import scripts
import os
import dill
from analysis import chicago, cad

start_date = datetime.datetime(2011, 3, 1)
cutoff_day_number = 277
end_date = start_date + datetime.timedelta(days=cutoff_day_number + 480)
niter = 75

# estimate_kwargs = {
#     'ct': 1,
#     'cd': 0.02,
#     'frac_bg': 0.5,
# }

estimate_kwargs = {
    'ct': 1 / 10.,
    'cd': 400,
    'frac_bg': 0.5,
}

model_kwargs = {
    'parallel': True,
    # 'max_delta_t': 120, # set on each iteration
    'max_delta_t': 1200, # set on each iteration
    # 'max_delta_d': 500, # set on each iteration
    'max_delta_d': 1000, # set on each iteration
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': None,
                           'strict': False,
                           'min_tol': 1e-8},
    # 'estimation_function': lambda x, y: estimation.estimator_bowers_fixed_proportion_bg(x, y, **estimate_kwargs),
    'estimation_function': lambda x, y: estimation.estimator_exp_gaussian(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
}


## CHICAGO SOUTH SIDE

chic_south = models.ChicagoDivision.objects.get(name='South').mpoly
chic_central = models.ChicagoDivision.objects.get(name='Central').mpoly
chic_sw = models.ChicagoDivision.objects.get(name='Southwest').mpoly
chic_n = models.ChicagoDivision.objects.get(name='North').mpoly

## Load from database
# res, t0, cid = chicago.get_crimes_by_type(crime_type='burglary', start_date=start_date, end_date=end_date,
#                                           domain=chic_south)

## Load from file
with open(os.path.join(scripts.IN_DIR, 'chicago_south', 'burglary.pickle'), 'r') as f:
# with open(os.path.join(scripts.IN_DIR, 'chicago', 'burglary.pickle'), 'r') as f:
    res = dill.load(f)

## CAMDEN

## Load from database
# res, t0, cid = cad.get_crimes_by_type(nicl_type=3)  # burglary

training = res[res[:, 0] <= cutoff_day_number]

# sepp_isotropic = pp_models.SeppStochasticNnIsotropicTrigger(data=training, **model_kwargs)
# ps_isotropic = sepp_isotropic.train(niter=niter)

sepp_local = pp_models.LocalSeppDeterministicNn(data=training, **model_kwargs)
ps_local = sepp_local.train(niter=niter)

# sepp_det = pp_models.SeppDeterministicNn(data=training, **model_kwargs)
# ps_det = sepp_det.train(niter=niter)

# sepp_xy = pp_models.SeppStochasticNn(data=training, **model_kwargs)
# ps_xy = sepp_xy.train(niter=niter)

## standardise the data
# s = np.std(training, axis=0, ddof=1)

## to preserve scaling, use same std on both X and Y
# s[1] = s[2] = s[1:].mean()
# training_s = training / s
# training_s = training_s - training_s.mean(axis=0)

# model_kwargs_s = model_kwargs.copy()
# model_kwargs_s['max_delta_d'] = model_kwargs['max_delta_d'] / s[1]
# model_kwargs_s['max_delta_t'] = model_kwargs['max_delta_t'] / s[0]

# sepp_xy_s = pp_models.SeppStochasticNn(data=training_s, **model_kwargs_s)
# ps_xy_s = sepp_xy_s.train(niter=niter)
