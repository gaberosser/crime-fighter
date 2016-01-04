from database.models import ArealUnit
from shapely import geometry
import os
from database.populate import sql_quote
import json
from network import itn


def get_borough_polys(srid=27700):
    """
    Retrieve all the borough multipolygons from the local DB in shapely format
    :param srid:
    :return:
    """
    obj = ArealUnit()
    res = obj.select(where_dict={'type': sql_quote('london_borough')},
                     fields=('ST_AsGeoJSON(ST_Transform(mpoly, {0})) AS domain'.format(srid),
                             'name'),
                     convert_to_dict=False)
    out = {}
    for r in res:
        out[r[1]] = geometry.shape(json.loads(r[0]))
    return out

def generate_network_borough_regions(full_network,
                                     outdir,
                                     buffer=250,
                                     n_segments=8):
    """
    Automatically produce cropped network regions by borough
    :param full_network: An instance derived from StreetNet
    :param outdir: The output directory. If it does not exist, it will be created
    :param buffer: The buffer size in metres
    :param n_segments: The number of segments to use when buffering to approximate a circle
    :return:
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    boroughs = get_borough_polys()

    for name, b in boroughs.iteritems():
        this_net = full_network.within_boundary(b.buffer(buffer, n_segments))
        this_net.label_edges_within_boundary(b)
        outfile = os.path.join(outdir, '%s.net' % name.lower().replace(' ', '_'))
        this_net.save(outfile)
        outfile = os.path.join(outdir, '%s' % name.lower().replace(' ', '_'))
        this_net.save(outfile, fmt='shp')
        print "Saved %s network as %s" % (name, outfile)
