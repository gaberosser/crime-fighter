from point_process import models, estimation, plots
import datetime
from analysis import chicago
from analysis.spatial import create_spatial_grid, shapely_rectangle_from_vertices
from database.chicago import loader
from matplotlib import pyplot as plt
from plotting import spatial, utils
import numpy as np


if __name__ == '__main__':

    side_name = 'South'
    niter = 60  # number of SEPP iterations before convergence is assumed
    num_validation = 100  # number of predict - assess cycles
    start_date = datetime.datetime(2011, 3, 1)  # first date for which data are required
    end_date = datetime.datetime(2012, 3, 1)
    start_day_number = 366  # number of days (after start date) on which first prediction is made

    estimate_kwargs = {
        'ct': 0.1,
        'cd': 150,
        'frac_bg': None,
    }

    model_kwargs = {
        'parallel': True,
        'max_delta_t': 90,
        'max_delta_d': 500,
        'bg_kde_kwargs': {'number_nn': [100, 15],
                          'min_bandwidth': None,
                          'strict': False},
        'trigger_kde_kwargs': {'number_nn': 15,
                               'min_bandwidth': [1., 50., 50.],
                               'strict': False},
        'estimation_function': lambda x, y: estimation.estimator_exp_gaussian(x, y, **estimate_kwargs),
        'seed': 42,  # doesn't matter what this is, just want it fixed
        'remove_coincident_pairs': False
    }

    pred_include = ('full_static',)  # only use this method for prediction

    domain = chicago.get_chicago_side_polys()[side_name]
    data, t0, cid = chicago.get_crimes_by_type(start_date=start_date,
                                               end_date=end_date,
                                               domain=domain)

    r = models.SeppStochasticNnReflected(data=data, **model_kwargs)
    ps = r.train(niter=niter)

    # marginal triggering
    t = np.linspace(0, 90, 500)
    x = np.linspace(-500, 500, 500)
    zt = r.trigger_kde.marginal_pdf(t, dim=0, normed=False) / float(r.ndata)
    zti = np.cumsum(zt) * (t[1] - t[0])  # numerical integral
    zx = r.trigger_kde.marginal_pdf(x, dim=1, normed=False) / float(r.ndata)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, zt)
    ax2 = ax.twinx()
    ax2.plot(t, zti, 'r-')
    ax.set_xlabel('Time (days)')
    ax.set_yticks([])
    ax.set_ylabel('Triggering intensity', labelpad=5, fontsize=18)
    ax2.set_ylabel('Cumulative prob. of triggering a crime', labelpad=12, fontsize=18, color='r')
    [tl.set_fontsize(18) for tl in ax2.get_yticklabels()]
    [tl.set_color('r') for tl in ax2.get_yticklabels()]
    [tl.set_fontsize(18) for tl in ax.get_xticklabels()]
    ax_pos = [0.1, 0.125, 0.7, 0.8]
    ax.set_position(ax_pos)
    ax2.set_position(ax_pos)

    fig.savefig('trigger_t_cu.png', dpi=300)
    fig.savefig('trigger_t_cu.pdf')

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, zx)
    ax.set_xlabel('Distance (metres)', fontsize=18)
    ax.set_yticks([])
    ax.set_ylabel('Triggering intensity', labelpad=5, fontsize=18)
    [tl.set_fontsize(18) for tl in ax.get_xticklabels()]
    ax_pos = [0.1, 0.125, 0.8, 0.8]
    ax.set_position(ax_pos)
    fig.savefig('trigger_x.png', dpi=300)
    fig.savefig('trigger_x.pdf')

    net = loader.load_network('s')
    net = net.within_boundary(domain)

    # heatmaps
    # (1) continuous
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111)
    cm = utils.transparent_colour_map()
    net.plot_network(ax=ax)

    pred_t = 180.
    arr_fun = lambda x, y: plots.txy_to_cartesian_data_array(np.ones_like(x) * pred_t, x, y)
    pred_fun = lambda x, y: r.predict(arr_fun(x, y))
    xx, yy, zz = spatial.plot_surface_function_on_polygon(domain, pred_fun, ax=ax, fmax=0.98, cmap=cm, dx=50)
    ax.axis('off')
    plt.tight_layout()

    fig.savefig('continuous_heatmap.png', dpi=300)
    fig.savefig('continuous_heatmap.pdf', dpi=300)

    # (2) top 10 % grid squares
    ipolys, full_extents, full_grid_square = create_spatial_grid(domain, 200)
    grid_values = []
    for xmin, ymin, xmax, ymax in full_extents:
        i, j = np.where((xx >= xmin) & (xx < xmax) & (yy >= ymin) & (yy < ymax))
        grid_values.append(zz[i, j].sum())
    sort_idx = np.argsort(grid_values)[::-1]
    top_10_pct = sort_idx[:int(len(sort_idx) * 0.1)]
    top_10_pct_grids = [shapely_rectangle_from_vertices(*full_extents[i]) for i in top_10_pct]

    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111)
    net.plot_network(ax=ax)
    spatial.plot_shapely_geos(domain, ax=ax)
    spatial.plot_shapely_geos(top_10_pct_grids, ax=ax, fc='r', ec='none', lw=1.5, alpha=0.7)

    ax.axis('off')
    plt.tight_layout()

    fig.savefig('top_10_pct_grids.png', dpi=300)
    fig.savefig('top_10_pct_grids.pdf', dpi=300)