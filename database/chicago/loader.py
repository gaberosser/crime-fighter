from ..network_loader import OSMLoader
import consts


class ChicagoLoader(OSMLoader):
    srid = 2028
    net_dir = consts.NETWORK_DIR


def load_network(region='s'):
    assert region in [t.lower() for t in consts.ABBREVIATED_REGIONS.values()], "Region code not recognised"
    a = [k for (k, v) in consts.ABBREVIATED_REGIONS.iteritems() if v.lower() == region.lower()][0]
    this_filename = "%s.osm" % consts.FILE_FRIENDLY_REGIONS[a]
    obj = ChicagoLoader(raw_filename=this_filename)
    return obj.load()