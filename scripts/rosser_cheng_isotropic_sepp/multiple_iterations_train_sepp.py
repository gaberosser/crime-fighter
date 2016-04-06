from point_process import simulate, estimation, models
from data.models import CartesianSpaceTimeData
from network.simulate import create_grid_network
from scripts import OUT_DIR
import os
import dill


if __name__ == '__main__':
    subdir = os.path.join(OUT_DIR, 'anisotropy_simulation_study', 'manhattan', 'sepp')
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    niter = 4
    domain_extent = [0., 0., 5000., 5000.]
    row_space = 100.
    col_spacings = [100., 200., 400., 800.][::-1]

    sim_t_total = 500.
    sim_num_to_prune = 400
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
                               'min_bandwidth': [0, 5., 5.],  # NB required?
                               'strict': False},
        'estimation_function': lambda x, y: estimation.estimator_exp_gaussian(x, y, **estimate_kwargs),
        'seed': 42,  # doesn't matter what this is, just want it fixed
        'remove_coincident_pairs': False
    }

    for col_space in col_spacings:
        print "row_space: %d, col_space: %d" % (row_space, col_space)

        # store containers
        data = []
        sepp = []

        for i in range(niter):
            net = create_grid_network(domain_extent, row_space, col_space)
            c = simulate.NetworkHomogBgExponentialGaussianTrig(net)
            c.trigger_params['sigma'] = 50.
            c.run(t_total=sim_t_total, num_to_prune=sim_num_to_prune)
            data.append(c.data)

            xy = c.data.space.to_cartesian()
            txy = c.data.time.adddim(xy, type=CartesianSpaceTimeData)
            r = models.SeppStochasticNnReflected(data=txy, **model_kwargs)
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

        fn = 'simulated_data_square_domain_network_row_%d_col_%d.dill' % (row_space, col_space)
        with open(os.path.join(subdir, fn), 'wb') as f:
            dill.dump(out, f)