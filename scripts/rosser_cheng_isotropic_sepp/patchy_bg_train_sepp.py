from point_process import simulate, estimation, models
from data.models import CartesianSpaceTimeData
from network.simulate import create_grid_network
from scripts import OUT_DIR
import numpy as np
import os
import dill


if __name__ == '__main__':
    subdir = os.path.join(OUT_DIR, 'anisotropy_simulation_study', 'manhattan', 'sepp')
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    niter = 10
    domain_extent = [0., 0., 5000., 5000.]
    col_spacings = [100., 200., 400., 800.]
    row_space = 100.

    sim_t_total = 1000.
    sim_num_to_prune = 400
    sim_bg_sigma = 100.

    train_kwargs = {
        'niter': 100,
    }

    estimate_kwargs = {
        'ct': 0.1,
        'cd': 50.,
        'frac_bg': None,
    }

    model_kwargs = {
        'parallel': True,
        'max_delta_t': 90, # set on each iteration
        'max_delta_d': 500, # set on each iteration
        'bg_kde_kwargs': {'number_nn': [100, 15],
                          'min_bandwidth': None,
                          'strict': False},
        'trigger_kde_kwargs': {'number_nn': 15,
                               'min_bandwidth': None,
                               'strict': False},
        'estimation_function': lambda x, y: estimation.estimator_exp_gaussian(x, y, **estimate_kwargs),
        'seed': 42,  # doesn't matter what this is, just want it fixed
        'remove_coincident_pairs': False
    }

    for col_space in col_spacings:
        print "row_space: %d, col_space: %d" % (row_space, col_space)

        #  generate simulation parameters
        bg_params = []
        x_bg = np.arange(domain_extent[0], domain_extent[2], col_space)
        y_bg = np.arange(domain_extent[1], domain_extent[3], row_space)
        n_bg = x_bg.size * y_bg.size
        for x in x_bg:
            for y in y_bg:
                bg_params.append({
                    'location': [x, y],
                    'intensity': 2. / n_bg,
                    'sigma': [sim_bg_sigma, sim_bg_sigma]
                })

        # store containers
        data = []
        sepp = []

        for i in range(niter):
            c = simulate.PatchyGaussianSumBackground(bg_params=bg_params)
            c.trigger_params['sigma'] = [50., 50.]  # TODO: change this, too?
            c.run(t_total=sim_t_total, num_to_prune=sim_num_to_prune)
            data.append(c.data)
            r = models.SeppStochasticNnReflected(data=c.data, **model_kwargs)
            _ = r.train(**train_kwargs)
            sepp.append(r)

        out = {
            'row_spacing': row_space,
            'col_spacing': col_space,
            'domain_extent': domain_extent,
            'sim_t_total': sim_t_total,
            'sim_num_to_prune': sim_num_to_prune,
            'train_kwargs': train_kwargs,
            'model_kwargs': model_kwargs,
            'data': data,
            'sepp': sepp,
        }

        fn = 'simulated_data_patchy_bg_row_%d_col_%d.dill' % (row_space, col_space)
        with open(os.path.join(subdir, fn), 'wb') as f:
            dill.dump(out, f)