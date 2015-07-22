__author__ = 'gabriel'
import datetime
from analysis import cad, spatial, chicago
from point_process import models as pp_models, estimation, validate, plots as pp_plotting
from point_process import simulate
from database import models
from validation import hotspot, validation
import copy
import numpy as np
from rpy2 import robjects, rinterface
import csv
import os
import settings

# T0 = 40603
T0 = 0
INITIAL_CUTOFF = 212
# DATA_CSV_DIR = os.path.join(settings.DATA_DIR, 'chicago', 'monsuru_data')

def get_chicago_polys(as_shapely=True):
    sides = models.ChicagoDivision.objects.filter(type='chicago_side')
    res = {}
    for s in sides:
        if as_shapely:
            res[s.name] = spatial.geodjango_to_shapely(s.mpoly.simplify())
        else:
            res[s.name] = s.mpoly.simplify()
    return res


def create_roc_grid(poly, grid_length=250):
    ipoly, fpoly, full = spatial.create_spatial_grid(poly, grid_length)
    return fpoly


def simulate_data(bg_intensity=10., trigger_intensity=0.2, num_results=4000):
    n_prune_min = 1000

    # estimate run time
    t = 1.1 * float(num_results + 2 * n_prune_min) / bg_intensity * (1 - trigger_intensity)

    c = simulate.PatchyGaussianSumBackground()

    c.bg_params[0]['location'] = [-1000., -1000.]
    c.bg_params[1]['location'] = [-1000., 1000.]
    c.bg_params[2]['location'] = [1000., -1000.]
    c.bg_params[3]['location'] = [1000., 1000.]
    c.bg_params[0]['sigma'] = [100, 500]
    c.bg_params[1]['sigma'] = [500, 500]
    c.bg_params[2]['sigma'] = [500, 500]
    c.bg_params[3]['sigma'] = [500, 100]
    for i in range(4):
        c.bg_params[i]['intensity'] = bg_intensity / 4.

    c.trigger_params['sigma'] = [20, 10]
    c.trigger_params['intensity'] = trigger_intensity
    c.t_total = t
    c.num_to_prune = n_prune_min
    c.run()

    while c.ndata < num_results:
        print "Too few results, increasing total time..."
        t *= 1.1
        c.run()

    # prune final results
    n_prune = int((c.ndata - num_results) / 2)
    c.prune_and_relabel(n_prune)

    data = c.data[:num_results, :3]
    return c, data


def simulate_data_symm(bg_intensity=10., trigger_intensity=0.2, num_results=4000):
    n_prune_min = 1000

    # estimate run time
    t = 1.1 * float(num_results + 2 * n_prune_min) / bg_intensity * (1 - trigger_intensity)

    c = simulate.PatchyGaussianSumBackground()

    c.bg_params[0]['location'] = [-1000., -1000.]
    c.bg_params[1]['location'] = [-1000., 1000.]
    c.bg_params[2]['location'] = [1000., -1000.]
    c.bg_params[3]['location'] = [1000., 1000.]
    c.bg_params[0]['sigma'] = [500, 500]
    c.bg_params[1]['sigma'] = [500, 500]
    c.bg_params[2]['sigma'] = [500, 500]
    c.bg_params[3]['sigma'] = [500, 500]
    for i in range(4):
        c.bg_params[i]['intensity'] = bg_intensity / 4.

    c.trigger_params['sigma'] = [10, 10]
    c.trigger_params['intensity'] = trigger_intensity
    c.t_total = t
    c.num_to_prune = n_prune_min
    c.run()

    while c.ndata < num_results:
        print "Too few results, increasing total time..."
        t *= 1.1
        c.run()

    # prune final results
    n_prune = int((c.ndata - num_results) / 2)
    c.prune_and_relabel(n_prune)

    data = c.data[:num_results, :3]
    return c, data


def get_chicago_data(primary_types=None, domain=None):
    start_date = datetime.date(2011, 3, 1)
    end_date = datetime.date(2012, 1, 6)
    # if domain is None:
    #     domain = models.ChicagoDivision.objects.get(name='South').mpoly.simplify()
    if primary_types is None:
        primary_types = (
            'burglary',
            'assault',
            'motor vehicle theft'
        )

    data = {}
    for pt in primary_types:
        key = pt.replace(' ', '_')
        data[key] = chicago.get_crimes_by_type(pt, start_date=start_date,
                                               end_date=end_date,
                                               domain=domain,
                                               convert_dates=True)
    return data


def apply_historic_kde(data,
                       data_index,
                       domain,
                       grid_squares=None,
                       num_sample_points=10,
                       time_window=60):
    ### Historic spatial KDE (Scott bandwidth) with integration sampling
    sk = hotspot.SKernelHistoric(time_window)
    vb = validation.ValidationIntegration(data,
                                          model=sk,
                                          data_index=data_index,
                                          spatial_domain=domain,
                                          cutoff_t=INITIAL_CUTOFF + T0)
    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True)
    return res


def apply_historic_kde_variable_bandwidth(data,
                                          data_index,
                                          domain,
                                          grid_squares=None,
                                          num_nn=20,
                                          num_sample_points=10,
                                          time_window=60):
    sk = hotspot.SKernelHistoricVariableBandwidthNn(dt=time_window, nn=num_nn)
    vb = validation.ValidationIntegration(data,
                                          model=sk,
                                          data_index=data_index,
                                          spatial_domain=domain,
                                          cutoff_t=INITIAL_CUTOFF + T0)

    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True)
    return res


def apply_sepp_stochastic_nn(data,
                             data_index,
                             domain,
                             grid_squares=None,
                             max_t=90,
                             max_d=500,
                             num_nn=(100, 15),
                             niter_training=50,
                             num_sample_points=10,
                             seed=43):

    est_fun = lambda x, y: estimation.estimator_bowers_fixed_proportion_bg(x, y, ct=1, cd=10, frac_bg=0.5)
    trigger_kde_kwargs = {
        'strict': False,
        'number_nn': num_nn[-1]
    }
    bg_kde_kwargs = {
        'strict': True,
        'number_nn': num_nn
    }

    sepp = pp_models.SeppStochasticNn(data=data,
                                      max_delta_t=max_t,
                                      max_delta_d=max_d,
                                      seed=seed,
                                      estimation_function=est_fun,
                                      trigger_kde_kwargs=trigger_kde_kwargs,
                                      bg_kde_kwargs=bg_kde_kwargs)

    vb = validate.SeppValidationFixedModelIntegration(data=data,
                                                      model=sepp,
                                                      data_index=data_index,
                                                      spatial_domain=domain,
                                                      cutoff_t=INITIAL_CUTOFF + T0)

    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True,
                 train_kwargs={'niter': niter_training})
    return res


if __name__ == '__main__':

    niter = 100

    ## (1) Simulated data
    res_sepp = {}
    res_kde = {}

    c, data = simulate_data_symm(num_results=1500)
    nns = (
        [100, 10],
        [100, 15],
        [100, 20],
        [100, 30],
        [100, 40],
        [100, 50],
        [100, 100],
        [50, 15],
        [40, 15],
        [30, 15],
        [20, 15],
        [15, 15],
    )
    est_fun = lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=0.1, cd=50, frac_bg=0.5)
    # est_fun = lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=10, cd=0.05, frac_bg=0.5)

    for num_nn in nns:
        trigger_kde_kwargs = {
            'strict': False,
            'number_nn': num_nn[-1]
        }
        bg_kde_kwargs = {
            'strict': False,
            'number_nn': list(num_nn)
        }

        sepp = pp_models.SeppStochasticNn(data=data,
                                          max_delta_t=50,
                                          max_delta_d=500,
                                          seed=42,
                                          estimation_function=est_fun,
                                          trigger_kde_kwargs=trigger_kde_kwargs,
                                          bg_kde_kwargs=bg_kde_kwargs)
        sepp.train(niter=niter)
        res_sepp[tuple(num_nn)] = copy.deepcopy(sepp)

    sepp = pp_models.SeppStochasticPluginBandwidth(
        data = data,
        max_delta_t=50,
        max_delta_d=500,
        seed=42,
        estimation_function=est_fun,
    )
    sepp.train(niter=niter)
    res_sepp['plugin'] = copy.deepcopy(sepp)

    # trigger_kde_kwargs = {
    #     'bandwidths': [10., 20., 10.],
    # }
    # bg_kde_kwargs = {
    #     'bandwidths': [],
    # }
    # sepp = pp_models.SeppStochastic(
    #     data = data,
    #     max_delta_t=50,
    #     max_delta_d=500,
    #     seed=42,
    #     estimation_function=est_fun,
    #
    # )

    ## (2) Real data, Chicago North

    res_chic_n = {}

    domain = get_chicago_polys()['South']
    tmp = get_chicago_data(domain=domain)
    data, t0, cid = tmp['burglary']

    est_fun = lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=0.1, cd=50, frac_bg=0.8)

    for num_nn in nns:
        trigger_kde_kwargs = {
            'strict': False,
            'number_nn': num_nn[-1]
        }
        bg_kde_kwargs = {
            'strict': False,
            'number_nn': list(num_nn)
        }

        sepp = pp_models.SeppStochasticNn(data=data,
                                          max_delta_t=90,
                                          max_delta_d=500,
                                          seed=42,
                                          estimation_function=est_fun,
                                          trigger_kde_kwargs=trigger_kde_kwargs,
                                          bg_kde_kwargs=bg_kde_kwargs)
        sepp.train(niter=niter)
        res_chic_n[tuple(num_nn)] = copy.deepcopy(sepp)

    trigger_kde_kwargs = {
        'strict': False,
    }
    bg_kde_kwargs = {
        'strict': True,
    }
    sepp = pp_models.SeppStochasticNn(data=data,
                                      max_delta_t=90,
                                      max_delta_d=500,
                                      seed=42,
                                      estimation_function=est_fun,
                                      trigger_kde_kwargs=trigger_kde_kwargs,
                                      bg_kde_kwargs=bg_kde_kwargs)
    sepp.train(niter=niter)
    res_chic_n['plugin'] = copy.deepcopy(sepp)