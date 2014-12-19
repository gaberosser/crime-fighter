__author__ = 'gabriel'
from analysis import cad
from point_process import models, estimation, validate
import datetime

# start_date is the FIRST DAY OF THE PREDICTION
start_date = datetime.datetime(2011, 9, 28)
estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 300,
    'bg_kde_kwargs': {'number_nn': [101, 16]},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 30, 30]},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs)
}
niter = 50

poly = cad.get_camden_region()

# load grid and create ROC for use in predictions
## TODO!

num_validation = 100

# end date is the last date retrieved from the database of crimes
# have to cast this to a date since the addition operation automatically produces a datetime
end_date = (start_date + datetime.timedelta(days=num_validation - 1)).date()

kinds = ['burglary', 'shoplifting', 'violence']

sepp_objs = {}
v_objs = {}

for k in kinds:

    data, t0 = cad.get_crimes_from_dump('monsuru_cad_%s' % k)
    # filter: day 210 is 27/9/2011, so use everything LESS THAN 211
    training_data = data[data[:, 0] < 211.]

    # train a model
    r = models.SeppStochasticNn(data=training_data, **model_kwargs)
    r.train(niter=niter)

    sepp_objs[k] = r
    vb = validate.SeppValidationPredefinedModel(data=data,
                                                model=r,
                                                spatial_domain=poly,
                                                cutoff_t=211)
    vb.set_grid(250) ## FIXME: pass in SpatialRoc created earlier.
    v_objs[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True)
