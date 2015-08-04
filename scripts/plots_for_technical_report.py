__author__ = 'gabriel'
from point_process import models as pp_models, plots as pp_plotting, estimation, validate, simulate
from validation import roc
from analysis import cad, chicago
from matplotlib import pyplot as plt
from database import osm


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
    polys = chicago.get_chicago_polys()
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
    polys = chicago.get_chicago_polys()
    nw = polys['Northwest']
    r = pp_models.SeppStochasticNn.from_pickle('/home/gabriel/data/chicago_northwest/burglary/vary_num_nn/nn_100_15.pickle')

    pp_plotting.multiplots(r)
    pp_plotting.prediction_heatmap(r, int(r.data.time.toarray()[-1]) + 1.0,
                                   poly=nw,
                                   dx=30,
                                   fmax=0.99,
                                   cmap='Reds')
    plt.axis('equal')