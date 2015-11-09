__author__ = 'gabriel'
import datetime
from analysis import cad, spatial, chicago
from point_process import models as pp_models, estimation, validate, plots as pp_plotting
from database import models
from validation import hotspot, validation
import numpy as np
from rpy2 import robjects, rinterface
import csv
import os
import settings


INITIAL_CUTOFF = 212
DATA_CSV_DIR = os.path.join(settings.DATA_DIR, 'chicago', 'monsuru_data')


estimate_kwargs = {
    'ct': 1,
    'cd': 0.02,
    'frac_bg': 0.5
    # 'frac_bg': None
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 300,
    'bg_kde_kwargs': {'number_nn': [101, 16], 'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 30, 30],
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers_fixed_proportion_bg(x, y, **estimate_kwargs)
}
niter = 150

poly = cad.get_camden_region()

# load grid and create ROC for use in predictions
# qset = models.Division.objects.filter(type='monsuru_250m_grid')
# qset = sorted(qset, key=lambda x:int(x.name))
# grid_squares = [t.mpoly[0] for t in qset]

num_validation = 100
num_sample_points = 30

pp_class=pp_models.SeppStochasticNn

# end date is the last date retrieved from the database of crimes
# have to cast this to a date since the addition operation automatically produces a datetime
# end_date = (start_date + datetime.timedelta(days=num_validation - 1)).date()

# kinds = ['burglary', 'shoplifting', 'violence']
kinds = ['burglary', ]

sepp_objs = {}
model_objs = {}
res = {}
vb_objs = {}
data_dict = {}
cid_dict = {}

for k in kinds:

    data, t0, cid = cad.get_crimes_from_dump('monsuru_cad_%s' % k)
    # filter: day 210 is 27/9/2011, so use everything LESS THAN 211

    ### SeppValidationFixedModel with integration ROC sampling

    b_sepp = True
    sepp = pp_class(data=data, **model_kwargs)
    vb = validate.SeppValidationFixedModelIntegration(data=data,
                                           model=sepp,
                                           data_index=cid,
                                           spatial_domain=poly,
                                           cutoff_t=INITIAL_CUTOFF,
                                           )

    vb.set_sample_units(250, num_sample_points)
    res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True,
                    train_kwargs={'niter': niter})


    sepp_objs[k] = vb.model