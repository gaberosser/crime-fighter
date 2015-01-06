__author__ = 'gabriel'
from analysis import cad
from point_process import models as pp_models, estimation, validate, plotting as pp_plotting
from database import models
import datetime
from matplotlib import pyplot as plt
import numpy as np
from rpy2 import robjects, rinterface

# start_date is the FIRST DAY OF THE PREDICTION
start_date = datetime.datetime(2011, 9, 28)
estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 300,
    'bg_kde_kwargs': {'number_nn': [101, 16], 'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 30, 30],
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs)
}
niter = 50

poly = cad.get_camden_region()

# load grid and create ROC for use in predictions
qset = models.Division.objects.filter(type='monsuru_250m_grid')
qset = sorted(qset, key=lambda x:int(x.name))
grid_squares = [t.mpoly[0] for t in qset]

num_validation = 100

# end date is the last date retrieved from the database of crimes
# have to cast this to a date since the addition operation automatically produces a datetime
end_date = (start_date + datetime.timedelta(days=num_validation - 1)).date()

kinds = ['burglary', 'shoplifting', 'violence']

sepp_objs = {}
res = {}
vb_objs = {}
data_dict = {}

for k in kinds:

    data, t0 = cad.get_crimes_from_dump('monsuru_cad_%s' % k)
    # filter: day 210 is 27/9/2011, so use everything LESS THAN 211
    training_data = data[data[:, 0] < 211.]

    # train a model
    r = pp_models.SeppStochasticNn(data=training_data, **model_kwargs)
    r.train(niter=niter)

    sepp_objs[k] = r
    vb = validate.SeppValidationPredefinedModel(data=data,
                                                model=r,
                                                spatial_domain=poly,
                                                cutoff_t=211)
    vb.set_grid(grid_squares)
    res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True)
    vb_objs[k] = vb
    data_dict[k] = data

# write data to files
outfile = 'sepp_assess.Rdata'
var_names = []
for k in kinds:
    # ranking
    var_name = 'grid_rank_%s' % k
    # need to add 1 to all rankings as Monsuru's IDs are one-indexed and mine are zero-indexed
    r_vec = robjects.IntVector(res[k]['full_static']['prediction_rank'].flatten() + 1)
    r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
    rinterface.globalenv[var_name] = r_mat
    var_names.append(var_name)

    # hit rate by crime count
    var_name = 'crime_count_%s' % k
    r_vec = robjects.IntVector(res[k]['full_static']['cumulative_crime_count'].flatten())
    r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
    rinterface.globalenv[var_name] = r_mat
    var_names.append(var_name)

robjects.r.save(*var_names, file=outfile)

# plots

for k in kinds:
    pp_plotting.validation_multiplot(res[k])