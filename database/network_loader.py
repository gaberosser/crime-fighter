import os
from network import osm, itn


class NetworkLoader(object):
    net_class = None
    net_dir = None
    srid = None

    @staticmethod
    def net_data_reader(raw_filename):
        raise NotImplementedError

    def __init__(self, raw_filename):
        """
        Base class for network loader
        :param net_class:
        :param net_data_class:
        :param net_dir:
        """
        self.raw_filename = os.path.join(self.net_dir, raw_filename)
        self.processed_filename = "%s.net" % os.path.splitext(self.raw_filename)[0]

    def convert_from_raw(self, fmt='pickle'):
        data_obj = self.net_data_reader(self.raw_filename)
        net_obj = self.net_class.from_data_structure(data_obj, srid=self.srid)
        net_obj.save(self.processed_filename, fmt=fmt)
        return net_obj

    def load(self):
        if not os.path.isfile(self.processed_filename):
            print "Processed file not found, generating now at %s" % self.processed_filename
            return self.convert_from_raw()
        else:
            print "Found processed file %s" % self.processed_filename
            return self.net_class.from_pickle(self.processed_filename)


class OSMLoader(NetworkLoader):
    net_class = osm.OSMStreetNet
    net_data_reader = osm.read_data

    @staticmethod
    def net_data_reader(raw_filename):
        return osm.read_data(raw_filename)


class ITNLoader(NetworkLoader):
    net_class = itn.ITNStreetNet
    net_data_reader = itn.read_gml

    @staticmethod
    def net_data_reader(raw_filename):
        return itn.read_gml(raw_filename)
