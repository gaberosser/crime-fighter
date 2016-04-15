from network.itn import read_gml, ITNStreetNet
from data import models
import os
import pickle
import bisect
from shapely import geometry
import settings
import numpy as np
from matplotlib import pyplot as plt
from validation import validation, hotspot, roc
import plotting.spatial
from analysis.spatial import shapely_rectangle_from_vertices
from network import plots as net_plots
from scripts import OUT_DIR, IN_DIR

INITAL_CUTOFF = 211

def camden_boundary():
    with open(os.path.join(settings.DATA_DIR, 'camden', 'boundary', 'boundary_line.pickle'), 'r') as f:
        xy = pickle.load(f)
        camden = geometry.Polygon(zip(*xy))
    return camden


def camden_safer_neighbourhoods():
    pass


def network_from_pickle():
    infile = os.path.join(IN_DIR, 'networks', 'camden_clipped.net')
    itn_net_reduced = ITNStreetNet.from_pickle(infile)
    return itn_net_reduced


def load_network_and_pickle():
    # test dataset is in a directory in the data directory called 'network_data'
    this_dir = os.path.join(settings.DATA_DIR, 'network_data')
    IN_FILE = os.path.join(this_dir, 'mastermap-itn_544003_0_camden_buff2000.gml')
    test_data = read_gml(IN_FILE)
    itn_net = ITNStreetNet.from_data_structure(test_data)

    camden = camden_boundary()

    itn_net_reduced = itn_net.within_boundary(camden.buffer(100))
    outdir = os.path.join(IN_DIR, 'networks')
    itn_net.save(os.path.join(outdir, 'camden.net'))
    itn_net_reduced.save(os.path.join(outdir, 'camden_clipped.net'))


def camden_network_plot():

    itn_net_reduced = network_from_pickle()

    itn_net_reduced.plot_network()
    camden = camden_boundary()
    plotting.spatial.plot_shapely_geos(camden, ec='r', fc='none', lw=3)


def jiggle_points(data):

    snapped = []
    free = []

    for t in data:
        if (t[1] % 25 == 0) and (t[2] % 25 == 0):
            snapped.append(t)
        else:
            free.append(t)

    snapped = np.array(snapped)
    free = np.array(free)

    snapped_jiggled = []
    for t in snapped:
        snapped_jiggled.append(t.copy())
        snapped_jiggled[-1][1:] += np.random.rand(2) * 250 - 125

    return snapped, snapped_jiggled, free


def snap_to_network(itn_net, data):
    snapped, snapped_jiggled, free = jiggle_points(data)
    free = models.CartesianSpaceTimeData(free)
    snapped_jiggled = models.CartesianSpaceTimeData(snapped_jiggled)

    space_net, failed = models.NetworkData.from_cartesian(itn_net, free.space, grid_size=50, return_failure_idx=True)
    keep = sorted(list(
        set(range(free.ndata)) - set(failed)
    ))
    time = free.time.getrows(keep)

    post_snap = time.adddim(
        space_net,
        type=models.NetworkSpaceTimeData
    )

    space_net_j, failed = models.NetworkData.from_cartesian(itn_net, snapped_jiggled.space, grid_size=50, return_failure_idx=True)
    keep = sorted(list(
        set(range(snapped_jiggled.ndata)) - set(failed)
    ))
    time_j = free.time.getrows(keep)
    post_snap_jiggled = time_j.adddim(
        space_net_j,
        type=models.NetworkSpaceTimeData
    )

    return free, post_snap, post_snap_jiggled


def network_heatmap(sample_points,
                    n_sample_points_per_unit,
                    fmax=0.95,
                    ax=None):
    """
    Plot showing spatial density on a network.
    :param sample_points: NetworkData array
    :param n_sample_points_per_unit: Array giving the number of sample points in each edge
    :return:
    """


    from network.plots import colorline

    norm = plt.Normalize(0., np.nanmax(z[~np.isinf(z)]) * fmax)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    i = 0
    for j, n in enumerate(n_sample_points_per_unit):
        # get x,y coords
        this_sample_points = sample_points.getrows(range(i, i + n))
        this_sample_points_xy = this_sample_points.to_cartesian()
        this_res = z[range(i, i + n)]

        colorline(this_sample_points_xy.toarray(0),
                  this_sample_points_xy.toarray(1),
                  z=this_res,
                  linewidth=5,
                  cmap=plt.get_cmap('coolwarm'),
                  alpha=0.9,
                  norm=norm
        )
        i += n

    points.graph.plot_network(edge_width=2, edge_inner_col='w')
    # plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])

    # ss = ((training.toarray(0) / max(training.toarray(0)))**2 * 200).astype(float)
    # plt.scatter(*training.space.to_cartesian().separate, s=ss, facecolors='none', edgecolors='k', alpha=0.6, zorder=11)

    plt.tight_layout()


def plot_compare_network_planar(net_v,
                                planar_v,
                                cutoff_t=None,
                                poly=None,
                                training=None,
                                testing=None,
                                cmap='Reds',
                                fmax=0.99,
                                bounds=None,
                                show_grid_ranking=False):
    """
    Produce plots comparing network and grid-based prediction methods.
    :param net_v: Validation object
    :param planar_v: Validation object
    :param cutoff_t: cutoff time, defaults to whatever is currently assigned
    :param poly: If supplied, plot the boundary poly
    :param training: If supplied, plot training data (i.e. past crimes)
    :param testing: If supplied, plot testing data (i.e. future crimes)
    :param cmap:
    :param fmax: Maximum colour level as a percentile
    :param bounds: If supplied, zoom plot to this region
    :param show_grid_ranking: If True, overlay numbers to show the order in which grid squares will be selected
    :return:
    """
    net_obj = net_v.data.graph

    # compute grid and net based prediction values
    net_z = net_v.model.predict(cutoff_t + 1., net_v.sample_points)
    planar_z = planar_v.predict(cutoff_t + 1.)
    # aggregate to grid - this is stolen directly from the ROC code
    planar_grid_values = []
    tally = 0
    for n in planar_v.roc.n_sample_point_per_unit:
        planar_grid_values.append(np.mean(planar_z[tally:tally + n]))
        tally += n
    planar_grid_values = np.array(planar_grid_values)

    # get max values
    idx = bisect.bisect_left(np.linspace(0, 1, net_z.size), fmax)
    net_vmax = sorted(net_z)[idx]
    idx = bisect.bisect_left(np.linspace(0, 1, planar_grid_values.size), fmax)
    planar_vmax = sorted(planar_grid_values)[idx]

    # get training data sizes if applicable
    # these are proportional to the time component
    if training:
        training_x = training.toarray(1)
        training_y = training.toarray(2)
        training_t = training.toarray(0)
        training_size = 500 * np.exp(-(cutoff_t - training_t) / 28.)

    # get testing data sizes if applicable
    if testing:
        testing_x = testing.toarray(1)
        testing_y = testing.toarray(2)
        testing_t = testing.toarray(0)
        testing_size = 500 * np.exp(-(testing_t - cutoff_t) / 3.)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 7))

    # left hand plot
    # plot basic network structure
    net_plots.plot_network_edge_lines(net_obj.lines_iter(), ax=axs[0])
    # overlay shaded grid
    grid = [shapely_rectangle_from_vertices(*t) for t in planar_v.roc.sample_units]
    plotting.spatial.plot_shaded_regions(grid,
                                         planar_grid_values,
                                         ax=axs[0],
                                         fmax=fmax,
                                         cmap=cmap,
                                         alpha=0.8,
                                         colorbar=False,
                                         scale_bar=None)
    if poly:
        plotting.spatial.plot_shapely_geos(poly, fc='none', ec='b', lw=1.5, ax=axs[0])
    if training:
        # zorder to elevate above the grid surface
        axs[0].scatter(training_x,
                       training_y,
                       edgecolors='b',
                       facecolors='none',
                       s=training_size,
                       lw=1.5,
                       zorder=10)
    if testing:
        # zorder to elevate above the grid surface
        axs[0].scatter(testing_x,
                       testing_y,
                       edgecolors='g',
                       facecolors='none',
                       s=testing_size,
                       lw=1.5,
                       zorder=10)



    # right hand plot
    axs[1].axis('off')
    # unfilled grid
    plotting.spatial.plot_shapely_geos(grid, fc='None', ec='k', alpha=0.8, ax=axs[1])
    # scatter density
    xy = net_v.sample_points.to_cartesian()
    axs[1].scatter(xy.toarray(0), xy.toarray(1), c=net_z,
                   s=20,
                   cmap=cmap,
                   edgecolor='none')
    net_plots.plot_network_edge_lines(net_obj.lines_iter(), ax=axs[1])

    if poly:
        plotting.spatial.plot_shapely_geos(poly, fc='none', ec='b', lw=1.5, ax=axs[1])
    if training:
        axs[1].scatter(training_x,
                       training_y,
                       edgecolors='b',
                       facecolors='none',
                       s=training_size,
                       lw=1.5)
    if testing:
        axs[1].scatter(testing_x,
                       testing_y,
                       edgecolors='g',
                       facecolors='none',
                       s=testing_size,
                       lw=1.5)

    plt.tight_layout(pad=0., h_pad=0., w_pad=0., rect=(0, 0, 1, 1))
    if bounds is not None:
        axs[0].axis(bounds)

    if show_grid_ranking:
        # compute ranking
        grid_ranking = np.argsort(planar_grid_values)
        for i in range(grid_ranking.size):
            c = grid[i].centroid
            axs[0].text(c.x,
                        c.y,
                        str(i + 1),
                        fontsize=20,
                        color='k',
                        zorder=11,
                        horizontalalignment='center',
                        verticalalignment='center',
                        clip_on=True)



if __name__ == "__main__":
    from analysis import cad
    from network import plots

    itn_net = network_from_pickle()

    n_test = 100  # number of testing days
    h = 200 # metres
    t_decay = 30 # days
    grid_length = 250
    n_samples = 20

    CRIME_TYPES = {
        'burglary': 3,
        'violence': 1,
        'shoplifting': 13,
    }

    crime_type = 3  # burglary
    boro_poly = camden_boundary()
    data, t0, cid = cad.get_crimes_by_type(nicl_type=crime_type)
    free, post_snap, post_snap_jiggled = snap_to_network(itn_net, data)
    # combine
    all_data = models.NetworkSpaceTimeData(np.vstack((post_snap_jiggled.data, post_snap.data)))

    psx, psy = all_data.space.to_cartesian().separate

    # network hotspot model
    sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=h, time_decay=t_decay)

    # get target points and test/training data
    vb = validation.NetworkValidationMean(all_data, sk, spatial_domain=None, include_predictions=True)
    vb.set_t_cutoff(INITAL_CUTOFF)
    vb.set_sample_units(None, n_samples)  # 2nd argument refers to interval between sample points

    targets = vb.sample_points
    z = sk.predict(INITAL_CUTOFF, targets)
    plots.network_lines_with_shaded_scatter_points(vb.sample_points, z, line_buffer=20)

    # vb_res = vb.run(1, n_iter=1)

        # compare with grid-based method with same parameters
        # cb_poly = camden_boundary()
        # data_txy = all_data.time.adddim(all_data.space.to_cartesian(), type=models.CartesianSpaceTimeData)
        # sk_planar = hotspot.STLinearSpaceExponentialTime(radius=h, mean_time=t_decay)
        # vb_planar = validation.ValidationIntegration(data_txy, sk_planar, spatial_domain=cb_poly, include_predictions=True)
        # vb_planar.set_t_cutoff(INITAL_CUTOFF)
        # vb_planar.set_sample_units(grid_length, n_samples)
        #
        # tic = time.time()
        # vb_res_planar = vb_planar.run(1, n_iter=n_test)
        # toc = time.time()
        # print toc - tic
        #
        # # compare with grid-based method using intersecting network segments to measure sample unit size
        # vb_planar_segment = validation.ValidationIntegrationByNetworkSegment(
        #     data_txy, sk_planar, spatial_domain=cb_poly, graph=itn_net
        # )
        # vb_planar_segment.set_t_cutoff(INITAL_CUTOFF)
        # vb_planar_segment.set_sample_units(grid_length, n_samples)
        #
        # tic = time.time()
        # vb_res_planar_segment = vb_planar_segment.run(1, n_iter=n_test)
        # toc = time.time()
        # print toc - tic
