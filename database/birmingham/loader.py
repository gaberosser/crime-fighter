from ..network_loader import ITNLoader
import consts


class BirminghamLoader(ITNLoader):
    srid = 27700
    net_dir = consts.NETWORK_DIR


def load_network():
    obj = BirminghamLoader('birmingham.gml')
    return obj.load()