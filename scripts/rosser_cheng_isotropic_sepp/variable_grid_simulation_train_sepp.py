from point_process import simulate, estimation, models, validate
from validation import evaluation
from data.models import CartesianSpaceTimeData
from analysis.spatial import shapely_rectangle_from_vertices
from network.simulate import create_grid_network
from scripts import OUT_DIR
import os
import dill
from matplotlib import pyplot as plt
import numpy as np


subdir = os.path.join(OUT_DIR, 'anisotropy_simulation_study', 'variable_grid_network')
row_space = 100.
col_spacings = [100., 200., 400., 800.]
domain_extent = [0., 0., 5000., 5000.]


def load_saved_data():
    fn = lambda c: os.path.join(subdir, 'simulated_data_square_domain_network_row_100_col_%d.dill') % c
    return dict([
        (c, dill.load(open(fn(c), 'rb'))) for c in col_spacings
    ])


def run_validation(all_data=None, cs=(100, 800), t0=300, sample_unit_size=50, n_validation=50):
    """
    Run a validation on the simulated data and pretrained SEPP model
    :param all_data:
    :param cs: Column spacings to use. Optional - default is [100, 800]
    :param t0: Initial cutoff
    :param sample_unit_size: Size of grid square to use for validation
    :param n_validation: Number of validation iterations to use
    :return:
    """
    if all_data is None:
        all_data = load_saved_data()
    domain = shapely_rectangle_from_vertices(*domain_extent)
    vb_res = {}
    for c in cs:
        try:
            the_sepp = all_data[c]['sepp'][0]
            the_data = all_data[c]['data'][0]
            txy = the_data.time.adddim(the_data.space.to_cartesian())
            vb = validate.SeppValidationPreTrainedModel(txy,
                                                        the_sepp,
                                                        include_predictions=False,
                                                        include=('full_static',))
            vb.set_t_cutoff(t0)
            vb.set_sample_units(sample_unit_size)
            vb_res[c] = vb.run(1, n_iter=n_validation)
        except Exception:
            pass
    return vb_res


def plot_triggering_fraction(all_data=None, trig_intensity=0.2):
    if all_data is None:
        all_data = load_saved_data()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xtm = ['%d' % t for t in col_spacings]
    frac_trig = []
    for c in col_spacings:
        the_sepp = all_data[c]['sepp'][0]
        frac_trig.append(1 - the_sepp.p.diagonal().sum() / the_sepp.ndata)
    # exact expression for the expected triggering fraction given the intensity
    true_frac = trig_intensity / (1 - trig_intensity)

    xl = 0.25 + np.arange(len(col_spacings))
    ax.bar(xl, frac_trig, width=0.5, fc='k', ec='none')
    xlim = [0, len(col_spacings)]
    ax.set_xlim(xlim)
    ax.set_ylim([0, 1])
    ax.plot(xlim, [true_frac, true_frac], 'r--', lw=2.5)
    ax.set_xticks(xl + 0.25)
    ax.set_xticklabels(xtm)
    ax.set_xlabel('Column spacing (m)')
    ax.set_ylabel('Inferred triggering fraction')


def plot_wilcox_test_hit_rate(vb_res1,
                              vb_res2,
                              ax=None,
                              sign_level=0.05,
                              max_cover=0.2,
                              min_difference=0.01):


    """
    Plot the mean hit rate vs coverage along with an overlay that indicates significant differences between methods.
    :param vb_res1: Dict from validation .run() call, method one (make this the one we're trying to prove)
    NB: choose the SEPP prediction type first, i.e. vb_res['full_static'] as pass that one here
    :param vb_res2: Result from validation .run() call, method two (make this the one we're trying to beat!)
    :param ax:
    :param sign_level: The one-tailed significance level
    :param max_cover:
    :param min_difference: The minimum magnitude of difference required before an overlay is plotted
    :return:
    """
    # convert sign level to two-tailed
    sign_level /= 2.

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    x1 = vb_res1['cumulative_area']
    x2 = vb_res2['cumulative_area']
    y1 = vb_res1['cumulative_crime']
    y2 = vb_res2['cumulative_crime']

    xm1, ym1 = evaluation.mean_hit_rate(x1, y1, n_covs=401)
    xm2, ym2 = evaluation.mean_hit_rate(x2, y2, n_covs=401)
    covs, pvals, eff, mdelta = evaluation.wilcoxon_comparison(x1, y1, x2, y2, max_coverage=max_cover)

    ax.plot(xm1, ym1, 'k')
    ax.plot(xm2, ym2, 'r')
    ax.fill_between(covs, 0, 1, (pvals < sign_level) & (eff == 1) & (mdelta >= min_difference),
                    facecolor='k', edgecolor='none', alpha=0.3, interpolate=True)
    ax.fill_between(covs, 0, 1, (pvals < sign_level) & (eff== -1) & (mdelta >= min_difference),
                    facecolor='r', edgecolor='none', alpha=0.3, interpolate=True)
    # Add darker shading for double min_difference
    ax.fill_between(covs, 0, 1, (pvals < sign_level) & (eff== 1) & (mdelta >= 2 * min_difference),
                    facecolor='k', edgecolor='none', alpha=0.3, interpolate=True)
    ax.fill_between(covs, 0, 1, (pvals < sign_level) & (eff == -1) & (mdelta >= 2 * min_difference),
                    facecolor='r', edgecolor='none', alpha=0.3, interpolate=True)

    ax.set_xlim([0, max_cover])

if __name__ == '__main__':

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    niter = 10


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
                               'min_bandwidth': None,
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