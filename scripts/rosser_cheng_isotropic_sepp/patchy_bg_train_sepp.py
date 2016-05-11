from point_process import simulate, estimation, models
from analysis.spatial import shapely_rectangle_from_vertices
from scripts.rosser_cheng_isotropic_sepp.ripleys_k_analysis import run_anisotropic_k
from scripts import OUT_DIR
import numpy as np
import os
import dill
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

subdir = os.path.join(OUT_DIR, 'anisotropy_simulation_study', 'patchy_background')
domain_extent = [0., 0., 5000., 5000.]
boundary = shapely_rectangle_from_vertices(*domain_extent)
col_spacings = [100., 200., 400., 800.]
row_space = 100.
sim_bg_sigma = 100.


def simulate_data_and_train():

    niter = 10

    sim_t_total = 1000.
    sim_num_to_prune = 400
    # sim_bg_sigma = 100.

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

        if not os.path.exists(subdir):
            os.makedirs(subdir)
        fn = 'simulated_data_patchy_bg_row_%d_col_%d.dill' % (row_space, col_space)
        with open(os.path.join(subdir, fn), 'wb') as f:
            dill.dump(out, f)


if __name__ == '__main__':
    # analyse results to produce figures

    dmax = 200.
    niter_rk = 50
    trigger_sigma = 50.

    # load previous results
    row_space = 100.
    sepp = {}
    data = {}
    bg_locs = {}

    for col_space in [col_spacings[0], col_spacings[-1]]:
        fn = 'simulated_data_patchy_bg_row_%d_col_%d.dill' % (row_space, col_space)
        with open(os.path.join(subdir, fn), 'rb') as f:
            res = dill.load(f)
        sepp[col_space] = res['sepp']
        data[col_space] = res['data']
        bg_locs[col_space] = []
        x_bg = np.arange(domain_extent[0], domain_extent[2], col_space)
        y_bg = np.arange(domain_extent[1], domain_extent[3], row_space)
        n_bg = x_bg.size * y_bg.size
        for x in x_bg:
            for y in y_bg:
                bg_locs[col_space].append([x, y])


    # run K ani routine
    k_obs = {}
    k_sim = {}
    for col_space in [col_spacings[0], col_spacings[-1]]:
        u, k_obs[col_space], k_sim[col_space] = run_anisotropic_k(data[col_space][0][:, 1:],
                                                                  boundary,
                                                                  dmax=dmax,
                                                                  nsim=niter_rk)

    # generate plots
    # 1) scatterplot of data

    # add an indicator of triggering extent
    loc = (domain_extent[0] + domain_extent[2] / 2., domain_extent[1] + domain_extent[3] / 2.)

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    plt.axis('equal')
    for i, col_space in enumerate([col_spacings[0], col_spacings[-1]]):
        # scatterplot
        axs[i].scatter(data[col_space][0][:, 1], data[col_space][0][:, 2], marker='o', facecolor='k', edgecolor='none',s=20, alpha=0.3)
        axs[i].set_aspect('equal')
        axs[i].set_xlim([domain_extent[0], domain_extent[2]])
        axs[i].set_ylim([domain_extent[1], domain_extent[3]])
        # add illustration of location of BG patches
        # for loc in bg_locs[col_space]:
        #     axs[i].add_patch(Circle(loc, sim_bg_sigma / 2., fc='none', ec='r', lw=1., alpha=0.4))
        axs[i].add_patch(Circle(loc, trigger_sigma, fc='r', ec='none'))
    axs[0].set_ylabel('Y (m)', fontsize=14)
    axs[0].set_xlabel('X (m)', fontsize=14)
    axs[1].set_xlabel('X (m)', fontsize=14)
    plt.tight_layout(pad=1.2)

    # 2) Ripley's K ani for the two
    styles = ('k--', 'r-', 'r--', 'k-')
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    for i, col_space in enumerate([col_spacings[0], col_spacings[-1]]):
        k_sim_min = k_sim[col_space].min(axis=0).mean(axis=1)
        k_sim_max = k_sim[col_space].max(axis=0).mean(axis=1)
        for j in range(4):
            axs[i].plot(u, k_obs[col_space][:, j], styles[j], lw=2.)
        axs[i].fill_between(u, k_sim_min, k_sim_max, color='k', alpha=0.3)
    axs[0].set_xlabel('Distance (m)', fontsize=14)
    axs[1].set_xlabel('Distance (m)', fontsize=14)
    axs[0].set_ylabel("Anisotropic Ripley's K", fontsize=14)
    fig.savefig('simulation_ripleys_k_ani.tiff', dpi=200)
    fig.savefig('simulation_ripleys_k_ani.png', dpi=300)
    fig.savefig('simulation_ripleys_k_ani.pdf', dpi=300)
    plt.tight_layout(pad=1.2)

    # 3) SEPP trigger results for the two
    from plot_trigger_ellipses import plot_spatial_ellipse_array
    fig = plt.figure()
    ax = fig.add_subplot(111)