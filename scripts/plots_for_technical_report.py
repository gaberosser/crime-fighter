__author__ = 'gabriel'
from point_process import models as pp_models, plots as pp_plotting, estimation, validate, simulate
from validation import roc
from kde import models as k_models
from analysis import cad, chicago
from matplotlib import pyplot as plt
from database import osm
import numpy as np


def camden_sampling_example():
    poly = cad.get_camden_region(as_shapely=True)
    r = cad.apply_point_process(max_delta_t=90,
                                sepp_class=pp_models.SeppStochasticNn,
                                only_new=False,
                                niter=100)

    pp_plotting.prediction_heatmap(r,
                                   int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=poly,
                                   dx=50,
                                   fmax=0.99,
                                   cmap='Reds')
    t = roc.RocGrid(poly=poly)
    t.set_sample_units(100)
    t.plot(ax=plt.gca(), show_sample_units=True, show_sample_points=False, show_prediction=False)

    pp_plotting.prediction_heatmap(r,
                                   int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=poly,
                                   dx=25,
                                   fmax=0.99,
                                   cmap='Reds')
    t = roc.RocGrid(poly=poly)
    t.set_sample_units(100)
    t.plot(ax=plt.gca(), show_sample_units=True, show_sample_points=True, show_prediction=False)

    ax = plt.gca()
    ax.set_xlim([527190, 530030])
    plt.draw()

    pp_plotting.prediction_heatmap(r,
                                   int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=poly,
                                   dx=25,
                                   fmax=0.99,
                                   cmap='Reds')
    t = roc.RocGridMean(poly=poly)
    t.set_sample_units(100, 20)
    t.plot(ax=plt.gca(), show_sample_units=True, show_sample_points=True, show_prediction=False)
    ax = plt.gca()
    ax.set_xlim([527190, 530030])
    plt.draw()


def simulation_mohler():
    c = simulate.MohlerSimulation()
    c.run()
    data = c.data[:, :3]
    max_delta_t = 100.
    max_delta_d = 1.
    initial_est = lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=0.1, cd=0.1)
    r = pp_models.SeppStochasticNn(data=data,
                                   max_delta_d=max_delta_d,
                                   max_delta_t=max_delta_t,
                                   estimation_function=initial_est)
    r.train(niter=50)
    pp_plotting.multiplots(r, c)


def cad_repeats():
    poly = cad.get_camden_region()
    o = osm.OsmRendererBase(poly, srid=27700)
    on_grid, off_grid_rpt, off_grid = cad.cad_spatial_repeat_analysis()  # burglary
    o.render(ax=plt.gca())

    on_grid, off_grid_rpt, off_grid = cad.cad_spatial_repeat_analysis()  # burglary
    o.render(ax=plt.gca())


def camden():
    # burglary
    poly = cad.get_camden_region(as_shapely=True)
    r = cad.apply_point_process(max_delta_t=90,
                                sepp_class=pp_models.SeppStochasticNn,
                                only_new=False,
                                niter=100)
    pp_plotting.multiplots(r)
    pp_plotting.prediction_heatmap(r, int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=poly,
                                   dx=30,
                                   fmax=0.99,
                                   cmap='Reds')
    plt.axis('equal')


def chicago_south():
    polys = chicago.get_chicago_side_polys()
    south = polys['South']
    nw = polys['Northwest']
    r = pp_models.SeppStochasticNn.from_pickle('/home/gabriel/data/chicago_south/burglary/vary_num_nn/nn_100_15.pickle')

    pp_plotting.multiplots(r)
    pp_plotting.prediction_heatmap(r, int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=south,
                                   dx=30,
                                   fmax=0.99,
                                   cmap='Reds')
    plt.axis('equal')


def chicago_northwest():
    polys = chicago.get_chicago_side_polys()
    nw = polys['Northwest']
    r = pp_models.SeppStochasticNn.from_pickle('/home/gabriel/data/chicago_northwest/burglary/vary_num_nn/nn_100_15.pickle')

    pp_plotting.multiplots(r)
    pp_plotting.prediction_heatmap(r, int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=nw,
                                   dx=30,
                                   fmax=0.99,
                                   cmap='Reds')
    plt.axis('equal')


def different_time_kernel():
    mu = 1.
    st = 1.
    t = np.linspace(-3, 5, 500)
    k1 = k_models.kernels.MultivariateNormal([mu], [st])
    k2 = k_models.kernels.SpaceTimeNormalReflective([mu], [st])

    z1 = k1.pdf(t)
    z2 = k2.pdf(t)

    t_refl = t[t<0][::-1]
    z_refl = k1.pdf(t_refl)

    import seaborn as sns
    sns.set_context("paper", font_scale=2.0)
    plt.figure(figsize=(8, 6))
    plt.plot(t, z1, 'k')
    plt.plot(t, z2, 'r-')
    plt.plot(-t_refl, z_refl, 'r--')
    plt.legend(('Normal', 'Reflective'))
    plt.xlabel('Time (days)')
    plt.ylabel('Density')

    c = simulate.MohlerSimulation()
    c.run()
    data = c.data[:, :3]
    max_delta_t = 100
    max_delta_d = 1.
    init_est_params = {
        'ct': 10,
        'cd': .05,
        'frac_bg': 0.5,
    }
    ra = pp_models.SeppStochasticNn(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t)
    rb = pp_models.SeppStochasticNnReflected(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t)
    ra.train(niter=20)
    rb.train(niter=20)

    t = np.linspace(-10, 60, 500)
    za = ra.trigger_kde.marginal_pdf(t, dim=0, normed=False) / float(ra.ndata)
    zb = rb.trigger_kde.marginal_pdf(t, dim=0, normed=False) / float(rb.ndata)
    w = c.trigger_params['time_decay']
    th = c.trigger_params['intensity']
    ztrue = th * w * np.exp(-w * t)
    ztrue[t<0] = 0

    plt.figure(figsize=(8, 6))
    plt.plot(t, za, 'k-')
    plt.plot(t, zb, 'r-')
    plt.plot(t, ztrue, 'k--')
    plt.legend(('Inferred, normal', 'Inferred, reflective', 'True'))
    plt.xlabel('Time (days)')
    plt.ylabel('Density')
