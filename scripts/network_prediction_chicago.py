from network.osm import read_data, OSMStreetNet
from data import models
import os
import datetime
from subprocess import call
import dill
import settings
import numpy as np
from matplotlib import pyplot as plt
from validation import validation, hotspot, roc
import plotting.spatial
from analysis import chicago
from database.chicago import consts
import analysis.spatial
from scripts import OUT_DIR

INITAL_CUTOFF = 211
START_DATE = datetime.date(2011, 3, 1)


def network_from_pickle(domain_name):
    infile = os.path.join(settings.DATA_DIR, 'osm_chicago', '%s_clipped.net' % domain_name)
    return OSMStreetNet.from_pickle(infile)


def load_network_and_pickle(domain, domain_name, buff=200):
    """
    Requires osmconvert to be installed
    sudo apt-get install osmctools
    :param domain: Shapely polygon or multipolygon
    :param domain_name: For file naming purposes
    :return:
    """
    outdir = os.path.join(settings.DATA_DIR, 'osm_chicago')
    illinois_data_path = os.path.join(settings.DATA_DIR, 'osm_pbf', 'illinois-latest.osm.pbf')
    out_file_stem = os.path.join(outdir, domain_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # 1: generate a poly file for osmconvert
    analysis.spatial.shapely_polygon_to_osm_poly(domain.buffer(buff), out_file_stem, srid=chicago.SRID)
    # 2: use osmconvert to extract that region
    call((
        'osmconvert',
        illinois_data_path,
        '--complete-ways',
        '-B=%s' % out_file_stem + '.poly',
        '-o=%s' % out_file_stem + '.osm'
    ))
    # 3: load that reduced network
    osmdata = read_data(out_file_stem + '.osm')
    o = OSMStreetNet.from_data_structure(osmdata, srid=chicago.SRID)
    # 4: reduce network, clipping lines
    o_clip = o.within_boundary(domain)
    # 5: save it
    o_clip.save(out_file_stem + '_clipped.net')

    return o_clip


def load_crimes(crime_type='burglary'):
    """
    Will need this method to load from pickle file IF we are running on Legion or in another environment with no
    DB.
    """
    pass


def snap_to_network(net, data):
    # copy data into appropriate type
    data = models.CartesianSpaceTimeData(data)
    # combine time dim with snapped network data
    res = data.time.adddim(
        models.NetworkData.from_cartesian(net, data.space, grid_size=50),
        type=models.NetworkSpaceTimeData
    )
    return res


def network_heatmap(model,
                    points,
                    points_per_edge,
                    t,
                    fmax=0.95,
                    ax=None):
    """
    Plot showing spatial density on a network.
    :param net: StreetNet instance
    :param model: Must have a pdf method that accepts a NetworkData object
    :param t: Time
    :param dr: Distance between network points
    :return:
    """
    from network.plots import colorline
    from network.utils import network_point_coverage
    z = model.predict(t, points)

    norm = plt.Normalize(0., np.nanmax(z[~np.isinf(z)]) * fmax)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    i = 0
    for j, n in enumerate(points_per_edge):
        # get x,y coords
        this_sample_points = points.getrows(range(i, i + n))
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





if __name__ == "__main__":

    DOMAIN = 'South'
    CRIME_TYPES = (
        'burglary',
        'assault',
        'motor vehicle theft',
    )

    n_test = 100  # number of testing days
    h = 200 # metres
    t_decay = 30 # days
    grid_length = 250
    n_samples = 20

    domain_long = consts.FILE_FRIENDLY_REGIONS[DOMAIN]
    domains = chicago.get_chicago_side_polys(as_shapely=True)
    poly = domains[DOMAIN]
    net = network_from_pickle(domain_long)

    for ct in CRIME_TYPES:

        data, t0, cid = chicago.get_crimes_by_type(crime_type=ct,
                                                   start_date=START_DATE,
                                                   end_date=START_DATE + datetime.timedelta(days=INITAL_CUTOFF + n_test - 1),
                                                   domain=poly)
        data_snapped = snap_to_network(net, data)

        sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=h, time_decay=t_decay)
        vb = validation.NetworkValidationMean(data_snapped, sk, spatial_domain=None, include_predictions=True)
        vb.set_t_cutoff(INITAL_CUTOFF)
        vb.set_sample_units(None, n_samples)  # 2nd argument refers to interval between sample points

        import time
        tic = time.time()
        vb_res = vb.run(1, n_iter=n_test)
        toc = time.time()
        print toc - tic

        # compare with grid-based method with same parameters
        sk_planar = hotspot.STLinearSpaceExponentialTime(radius=h, mean_time=t_decay)
        vb_planar = validation.ValidationIntegration(data, sk_planar, spatial_domain=poly, include_predictions=True)
        vb_planar.set_t_cutoff(INITAL_CUTOFF)
        vb_planar.set_sample_units(grid_length, n_samples)

        tic = time.time()
        vb_res_planar = vb_planar.run(1, n_iter=n_test)
        toc = time.time()
        print toc - tic


        # compare with grid-based method using intersecting network segments to measure sample unit size
        vb_planar_segment = validation.ValidationIntegrationByNetworkSegment(
            data, sk_planar, spatial_domain=poly, graph=net
        )
        vb_planar_segment.set_t_cutoff(INITAL_CUTOFF)
        vb_planar_segment.set_sample_units(grid_length, n_samples)

        tic = time.time()
        vb_res_planar_segment = vb_planar_segment.run(1, n_iter=n_test)
        toc = time.time()
        print toc - tic

        # dump to pickle file
        out = {
            'network_kde': vb_res,
            'planar_kde': vb_res_planar,
            'planar_kde_by_road_length': vb_res_planar_segment
        }
        outfile = os.path.join(OUT_DIR, 'network_vs_planar', '%s_%s.pickle' % (domain_long,
                                                                               ct.replace(' ', '_')))
        with open(outfile, 'w') as f:
            dill.dump(out, f)