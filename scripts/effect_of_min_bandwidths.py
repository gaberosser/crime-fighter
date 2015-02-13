__author__ = 'gabriel'
from analysis import cad, spatial
from point_process import models as pp_models, estimation, validate, plotting as pp_plotting
from database import models
import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
import dill

ROOT_DIR = '/home/gabriel/pickled_results'

# start_date is the FIRST DAY OF THE PREDICTION
start_date = datetime.datetime(2011, 12, 3)
# equivalent in number of days from t0 (1/3/2011)
start_day_number = 277
# start_day_number = 385
num_sample_points = 20

estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 400,
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,  # replace this on each iteration
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': None,  # replace this on each iteration
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
}

niter = 75
# niter = 25
num_validation = 120

min_t_bds = [0, 0.5, 1, 2]
min_d_bds = [0, 20, 50, 100]

tt, dd = np.meshgrid(min_t_bds, min_d_bds)

poly = cad.get_camden_region()
qset = models.Division.objects.filter(type='cad_250m_grid')
qset = sorted(qset, key=lambda x:int(x.name))
grid_squares = [t.mpoly[0] for t in qset]



# define crime types
crime_types = {
    'burglary': 3,
    'robbery': 5,
    'theft of vehicle': 6,
    'violence': 1,
}

try:
    for (name, n) in crime_types.items():
        data, t0, cid = cad.get_crimes_by_type(n)

        sepp_objs = {}
        vb_objs = {}
        res = {}

        # jiggle grid-snapped points
        # data = spatial.jiggle_on_grid_points(data, grid_squares)

        for t, d in zip(tt.flat, dd.flat):
            model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = [t, d, d]
            model_kwargs['bg_kde_kwargs']['min_bandwidth'] = [t, d, d]
            vb = validate.SeppValidationFixedModelIntegration(data=data,
                                                   pp_class=pp_models.SeppStochasticNn,
                                                   data_index=cid,
                                                   spatial_domain=poly,
                                                   cutoff_t=start_day_number,
                                                   model_kwargs=model_kwargs,
                                                   )

            vb.set_grid(grid_squares, num_sample_points)

            try:
                res[(t, d)] = vb.run(time_step=1, n_iter=num_validation, verbose=True,
                                train_kwargs={'niter': niter})
                sepp_objs[(t, d)] = vb.model
                vb_objs[(t, d)] = vb
            except Exception as exc:
                print exc
                res[(t, d)] = None
                sepp_objs[(t, d)] = None
                vb_objs[(t, d)] = None
        with open(os.path.join(ROOT_DIR, 'camden', name, 'sepp_obj.pickle'), 'w') as f:
            dill.dump(sepp_objs, f)
        with open(os.path.join(ROOT_DIR, 'camden', name, 'validation_obj.pickle'), 'w') as f:
            dill.dump(vb_objs, f)
        with open(os.path.join(ROOT_DIR, 'camden', name, 'validation.pickle'), 'w') as f:
            dill.dump(res, f)

finally:
    f = open('/home/gabriel/signal/shut_me_down_goddamnit', 'w')
    f.close()

# shutdown PC (requires that script is run with sudo)
# os.system("shutdown now -h")