__author__ = 'gabriel'
from analysis import cad, chicago
from point_process import models as pp_models, estimation, validate
from database import models
import datetime
import numpy as np
import os
import dill
import io
from utils import shutdown_decorator

ROOT_DIR = '/home/gabriel/pickled_results'

num_sample_points = 20

estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 400,
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': [0.5, 20, 20],
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 20, 20],
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
}

niter = 75

## CAMDEN
@shutdown_decorator
def camden():

    start_day_numbers = [277, 307, 337, 367]  # number of days from t0 (1/3/2011)

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
        try:
            data, t0, cid = cad.get_crimes_by_type(n)
            t_upper = data[-1, 0]

            sepp_objs = {}
            vb_objs = {}
            res = {}

            for sdn in start_day_numbers:

                vb = validate.SeppValidationFixedModelIntegration(data=data,
                                                       pp_class=pp_models.SeppStochasticNn,
                                                       data_index=cid,
                                                       spatial_domain=poly,
                                                       cutoff_t=sdn,
                                                       model_kwargs=model_kwargs,
                                                       )

                vb.set_sample_units(grid_squares, num_sample_points)

                try:
                    res[sdn] = vb.run(time_step=1, t_upper=t_upper, verbose=True,
                                    train_kwargs={'niter': niter})
                    sepp_objs[sdn] = vb.model
                    vb_objs[sdn] = vb
                except Exception as exc:
                    print exc
                    res[sdn] = None
                    sepp_objs[sdn] = None
                    vb_objs[sdn] = None

            with open(os.path.join(ROOT_DIR, 'camden', 'model_ageing', name, 'sepp_obj.pickle'), 'w') as f:
                dill.dump(sepp_objs, f)
            with open(os.path.join(ROOT_DIR, 'camden', 'model_ageing', name, 'validation_obj.pickle'), 'w') as f:
                dill.dump(vb_objs, f)
            with open(os.path.join(ROOT_DIR, 'camden', 'model_ageing', name, 'validation.pickle'), 'w') as f:
                dill.dump(res, f)

        except Exception as exc:
            with open(os.path.join(ROOT_DIR, 'camden', 'model_ageing', name, 'errors'), 'a') as f:
                f.write(repr(exc))
                f.write('\n')


@shutdown_decorator
def run_chicago():

    start_date = datetime.datetime(2011, 3, 1)
    end_date = start_date + datetime.timedelta(days=277 + 480)
    start_day_numbers = [277 + 30 * i for i in range(18)]

    poly = chicago.compute_chicago_region()
    south = models.ChicagoDivision.objects.get(name='South').mpoly

    # define crime types
    crime_types = {
        'burglary': 'burglary',
        'robbery': 'robbery',
        'theft_of_vehicle': 'motor vehicle theft',
        'violence': 'assault',
    }

    for (name, pt) in crime_types.items():
        print "Crime type: %s" % name
        base_dir = os.path.join(ROOT_DIR, 'chicago', 'model_ageing', name)

        try:
            data, t0, cid = chicago.get_crimes_by_type(crime_type=pt,
                                                       start_date=start_date,
                                                       end_date=end_date,
                                                       domain=south)
            t_upper = data[-1, 0]

            sepp_objs = {}
            vb_objs = {}
            res = {}

            for sdn in start_day_numbers:

                vb = validate.SeppValidationFixedModelIntegration(data=data,
                                                                  pp_class=pp_models.SeppStochasticNn,
                                                                  data_index=cid,
                                                                  spatial_domain=south,
                                                                  cutoff_t=sdn,
                                                                  model_kwargs=model_kwargs,
                                                                  )

                vb.set_sample_units(250, num_sample_points)

                try:
                    res[sdn] = vb.run(time_step=1, t_upper=t_upper, verbose=True,
                                    train_kwargs={'niter': niter})
                    sepp_objs[sdn] = vb.model
                    vb_objs[sdn] = vb
                except Exception as exc:
                    print exc
                    res[sdn] = None
                    sepp_objs[sdn] = None
                    vb_objs[sdn] = None

            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

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


def analyse_chicago(coverage=0.2):

    crime_types = {
        'burglary': 'burglary',
        # 'robbery': 'robbery',
        # 'theft_of_vehicle': 'motor vehicle theft',
        # 'violence': 'assault',
    }

    pai = {}
    hr = {}
    hr30 = {}
    pai30 = {}

    for k in crime_types.keys():
        base_dir = os.path.join(ROOT_DIR, 'chicago', 'model_ageing', k)
        with open(os.path.join(base_dir, 'validation.pickle'), 'r') as f:
            vres = dill.load(f)
        this_pai = {}
        this_hr = {}
        this_hr30 = {}
        this_pai30 = {}
        for n in vres.keys():
            if vres[n] is None:
                continue
            this = vres[n]['full']
            x = this['cumulative_area'].mean(axis=0)
            idx = np.where(x >= coverage)[0][0]

            tmp_hr = this['cumulative_crime'][:, idx]
            tmp_pai = this['pai'][:, idx]
            not_nan_idx = ~np.isnan(tmp_hr)
            this_hr[n] = tmp_hr[not_nan_idx]
            this_pai[n] = tmp_pai[not_nan_idx]

            # split into chunks of approximately 30 days
            thr30 = {}
            tpai30 = {}
            l = len(x)
            start_idx = range(0, l, 30)
            for i in start_idx:
                thr = tmp_hr[i:i+30]
                thr = thr[~np.isnan(thr)]
                tpai = tmp_pai[i:i+30]
                tpai = tpai[~np.isnan(tpai)]
                thr30[n + i] = thr
                tpai30[n + i] = tpai
            this_hr30[n] = thr30
            this_pai30[n] = tpai30

        hr[k] = this_hr
        pai[k] = this_pai
        hr30[k] = this_hr30
        pai30[k] = this_pai30

    return hr, pai, hr30, pai30