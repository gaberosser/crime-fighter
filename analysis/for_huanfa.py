from validation import roc, hotspot

__author__ = 'gabriel'
import cad, chicago
from network import osm
import collections
import datetime
import shapefile
import os
from django.contrib.gis import geos
from settings import DATA_DIR
from database.chicago import consts
from data.models import NetworkData, NetworkSpaceTimeData, DataArray


def create_grid_centroid_hotspots_shapefile(grid_spacing=25):
    grid_spacing = 25  # distance between centroids in metres

    res, t0, cid = cad.get_crimes_by_type(nicl_type=range(1, 17))
    sk = hotspot.SKernelHistoricVariableBandwidthNn(dt=60, nn=15)

    # choose final date 2011/11/1 for training -> 60 days prior ends AFTER crossover to new non-snapped data
    days_offset = 245
    dt_pred = t0.date() + datetime.timedelta(days=days_offset)
    training = res[res[:, 0] <= days_offset]
    sk.train(training)

    # generate grid
    camden = cad.get_camden_region()
    # convert to buffered rectangular region
    sq_poly = camden.buffer(grid_spacing).envelope
    # lock to integer multiples of grid_spacing
    c = []
    for i in range(len(sq_poly.coords[0])):
        a = int(sq_poly.coords[0][i][0] / grid_spacing) * grid_spacing
        b = int(sq_poly.coords[0][i][1] / grid_spacing) * grid_spacing
        c.append((a, b))
    sq_poly = geos.Polygon(geos.LinearRing(c))

    r = roc.RocGrid(poly=sq_poly)
    print "Setting ROC grid..."
    r.set_sample_units(grid_spacing)
    print "Done."

    # prediction on grid
    print "Computing KDE prediction..."
    z = sk.predict(r.sample_points)
    print "Done."

    print "Writing shapefile..."
    out_shp = 'risk_surface_%s.shp' % dt_pred.strftime('%Y-%m-%d')
    w = shapefile.Writer(shapefile.POINT)
    w.field('risk', fieldType='N')
    for x, y in zip(r.sample_points, z.flat):
        w.point(int(x[0]), int(x[1]))
        w.record(risk=y)
    w.save(out_shp)
    print "Done."


def create_network_with_crime_counts(start_date=datetime.date(2011, 3, 1),
                                     domain_name='South'):

    # load network, count crimes in 6mo and 12mo window, output shapefile
    domains = chicago.get_chicago_side_polys()
    domain = domains[domain_name]

    end_date = start_date + datetime.timedelta(days=365)
    crime_types = (
        'THEFT',
        'BURGLARY',
        'HOMICIDE',
        'BATTERY',
        'ARSON',
        'MOTOR VEHICLE THEFT',
        'ASSAULT',
        'ROBBERY',
    )

    time_window_filters = {
        '6mo': lambda t: t <= 183,
        '12mo': lambda t: t <= 365,
    }

    # get crime data
    data, t0, cid = chicago.get_crimes_by_type(crime_type=crime_types,
                                               start_date=start_date,
                                               end_date=end_date,
                                               domain=domain)

    # get network
    osm_file = os.path.join(DATA_DIR, 'osm_chicago', '%s_clipped.net' % consts.FILE_FRIENDLY_REGIONS[domain_name])
    net = osm.OSMStreetNet.from_pickle(osm_file)

    # snap crime data to network with maximum distance cutoff
    netdata, failed = NetworkData.from_cartesian(net, data[:, 1:], radius=50, return_failure_idx=True)
    # get non-failed times
    idx = sorted(set(range(data.shape[0])) - set(failed))
    netdata = DataArray(data[idx, 0]).adddim(netdata, type=NetworkSpaceTimeData)

    # run over edges, count crimes in the two time windows
    filters = {}
    for filt_name, filt_func in time_window_filters.items():
        filters[filt_name] = filt_func(netdata.toarray(0)).astype(int)

    # add count attributes to all edges
    for e in net.edges():
        e.attrs['crimes_6mo'] = 0
        e.attrs['crimes_12mo'] = 0

    edge_counts = {}
    for i, t in enumerate(netdata.space.toarray()):
        t.edge.attrs['crimes_6mo'] += filters['6mo'][i]
        t.edge.attrs['crimes_12mo'] += filters['12mo'][i]

    net.save(consts.FILE_FRIENDLY_REGIONS[domain_name] + '_network_crime_counts', fmt='shp')


if __name__ == "__main__":
    pass