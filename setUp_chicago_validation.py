__author__ = 'gabriel'
import numpy as np
import datetime
from analysis import plotting, chicago
import dill
from point_process import estimation, models, validate
from database.models import ChicagoDivision


# start_date is the FIRST DAY OF THE PREDICTION
start_date = datetime.datetime(2013, 6, 1)
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

chic = chicago.compute_chicago_region()
southwest = ChicagoDivision.objects.get(name='Southwest').mpoly
south = ChicagoDivision.objects.get(name='South').mpoly
west = ChicagoDivision.objects.get(name='West').mpoly

southwest_buf = southwest.buffer(1500)
south_buf = south.buffer(1500)
west_buf = west.buffer(1500)

training_size = 120
num_validation = 30

# compute number of days in date range
pre_start_date = start_date - datetime.timedelta(days=training_size)
ndays = training_size + num_validation

# end date is the last date retrieved from the database of crimes
# have to cast this to a date since the addition operation automatically produces a datetime
end_date = (start_date + datetime.timedelta(days=num_validation - 1)).date()

res, t0 = chicago.get_crimes_by_type(
    crime_type='burglary',
    start_date=pre_start_date,
    end_date=end_date,
)

# train a model
training_data = res[res[:, 0] < training_size]
r = models.SeppStochasticNn(data=training_data, **model_kwargs)
r.train(niter=niter)

# centroid method

vb_centroid = {}
res_centroid = {}

vb_centroid_500 = validate.SeppValidationPredefinedModel(data=res,
                                            model=r,
                                            spatial_domain=south,
                                            cutoff_t=120)

vb_centroid_500.set_grid(500)
res_centroid[500] = vb_centroid_500.run(time_step=1, n_iter=30, verbose=True)
vb_centroid[500] = vb_centroid_500

vb_centroid_250 = validate.SeppValidationPredefinedModel(data=res,
                                            model=r,
                                            spatial_domain=south,
                                            cutoff_t=120)
vb_centroid_250.set_t_cutoff(120)
vb_centroid_250.set_grid(250)
res_centroid[250] = vb_centroid_250.run(time_step=1, n_iter=30, verbose=True)
vb_centroid[250] = vb_centroid_250

params = [
    (500, 10),
    (500, 50),
    (250, 10),
    (250, 50)
]

# sample point method without edge correction

vb_sample_points = {}
res_sample_points = {}

for t in params:
    vb_sample_points[t] = validate.SeppValidationPredefinedModelIntegration(data=res,
                                                           model=r,
                                                           spatial_domain=south,
                                                           cutoff_t=120)
    vb_sample_points[t].set_grid(t[0], t[1], respect_boundary=False)
    res_sample_points[t] = vb_sample_points[t].run(time_step=1, n_iter=30, verbose=True)

# sample point method with edge correction

vb_sample_points_ec = {}
res_sample_points_ec = {}

for t in params:
    vb_sample_points_ec[t] = validate.SeppValidationPredefinedModelIntegration(data=res,
                                                           model=r,
                                                           spatial_domain=south,
                                                           cutoff_t=120)
    vb_sample_points_ec[t].set_grid(t[0], t[1], respect_boundary=True)
    res_sample_points_ec[t] = vb_sample_points_ec[t].run(time_step=1, n_iter=30, verbose=True)


"""

this_res, vb = chicago.validate_point_process(
    start_date=start_date,
    training_size=120,
    num_validation=30,
    num_pp_iter=50,
    domain=chic,
    grid=500,
    n_sample_per_grid=10,
    model_kwargs=model_kwargs
)

res = {
    (500, 10): this_res
}

# expt 1: increase the number of sample points, keeping a grid of 500m side length

if True:

    # re-run with more sampling points
    n_samples = [5, 20, 30, 50]
    for n in n_samples:
        new_vb = copy.deepcopy(vb)
        new_vb.roc.set_grid(500, n_sample_per_grid=n)
        res[(500, n)] = new_vb.repeat_run(this_res)

# expt 2: increase the grid fineness, keeping 10 sample points per grid square

if True:

    grid_lengths = [400, 250, 100]
    for n in grid_lengths:
        new_vb = copy.deepcopy(vb)
        new_vb.roc.set_grid(n, n_sample_per_grid=10)
        res[(n, 10)] = new_vb.repeat_run(this_res)

"""