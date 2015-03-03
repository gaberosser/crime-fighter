__author__ = 'gabriel'
from analysis import cad, spatial
from point_process import models as pp_models, estimation, validate, plotting as pp_plotting
from database import models
import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
import dill
from scipy import stats
import io
from utils import shutdown_decorator

ROOT_DIR = '/home/gabriel/pickled_results'

# global parameters
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



# def test_fade_out(coverage=0.2, min_bandwidth=(0.5, 50), plot=False):
#     res = io.load_camden_validation_evaluation(coverage)
#     linregress_keys = ('slope', 'intercept', 'rval', 'pval', 'stderr')
#     out = {}
#     for k in res.keys():
#         # full portion
#
#         this_res = res[k][min_bandwidth]
#         n = this_res['hit_rate'].size
#         idx = ~np.isnan(this_res['hit_rate'])
#         t = np.arange(n)[idx]
#         hr = this_res['hit_rate'][idx]
#         pai = this_res['pai'][idx]
#         out[k] = {
#             'hit_rate': dict([(x, y) for x, y in zip(linregress_keys, stats.linregress(t, hr))]),
#             'pai': dict([(x, y) for x, y in zip(linregress_keys, stats.linregress(t, pai))]),
#         }
#
#         if plot:
#             fig = plt.figure(k)
#             ax = fig.add_subplot(111)
#             ax.plot(t, pai, 'ko')
#             ax.plot(t, out[k]['pai']['intercept'] + t * out[k]['pai']['slope'], 'k--')
#             ax.set_xlabel('Prediction day', fontsize=14)
#             ax.set_ylabel('PAI', fontsize=14)
#             ax.set_ylim([-0.02/coverage, 1.02/coverage])
#
#     return out


# @shutdown_decorator
def camden():

    # start_date is the FIRST DAY OF THE PREDICTION
    start_date = datetime.datetime(2011, 12, 3)
    # equivalent in number of days from t0 (1/3/2011)
    start_day_number = 277
    # start_day_number = 385

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
        'theft_of_vehicle': 6,
        'violence': 1,
    }

    for (name, n) in crime_types.items():
        print "Crime type: %s" % name
        base_dir = os.path.join(ROOT_DIR, 'camden', 'min_bandwidth', name)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

        try:
            data, t0, cid = cad.get_crimes_by_type(n)

            sepp_objs = {}
            vb_objs = {}
            res = {}

            # jiggle grid-snapped points
            # data = spatial.jiggle_on_grid_points(data, grid_squares)

            # check that num_validation iterations is feasible
            if start_day_number + num_validation - 1 > data[-1, 0]:
                this_num_validation = int(data[-1, 0]) - start_day_number + 1
                print "Can only do %d validation runs" % this_num_validation
            else:
                this_num_validation = num_validation

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
                    res[(t, d)] = vb.run(time_step=1, n_iter=this_num_validation, verbose=True,
                                    train_kwargs={'niter': niter})
                    sepp_objs[(t, d)] = vb.model
                    vb_objs[(t, d)] = vb
                except Exception as exc:
                    print exc
                    res[(t, d)] = None
                    sepp_objs[(t, d)] = None
                    vb_objs[(t, d)] = None
            with open(os.path.join(base_dir, 'sepp_obj.pickle'), 'w') as f:
                dill.dump(sepp_objs, f)
            with open(os.path.join(base_dir, 'validation_obj.pickle'), 'w') as f:
                dill.dump(vb_objs, f)
            with open(os.path.join(base_dir, 'validation.pickle'), 'w') as f:
                dill.dump(res, f)

        except Exception as exc:
            with open(os.path.join(base_dir, 'errors'), 'a') as f:
                f.write(repr(exc))
                f.write('\n')
