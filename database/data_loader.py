__author__ = 'gabriel'
import csv
import pickle
from django.db import connection
import numpy as np


def datetime_to_days(t0, dt):
    delta = dt - t0
    return delta.total_seconds() / 86400.


class DataLoaderBase(object):

    idx_field = None

    def __init__(self, convert_dates=True, as_txy=True):
        self.convert_dates = convert_dates
        self.as_txy = as_txy
        self.t0 = None
        self.idx = None

    def load_raw_data(self, **kwargs):
        raise NotImplementedError()

    def process_raw_data(self, raw):
        return raw

    def date_conversion(self, processed):
        raise NotImplementedError()

    def set_idx_field(self, processed):
        if self.idx_field is not None:
            self.idx = [t.get(self.idx_field) for t in processed]
        else:
            self.idx = np.arange(len(processed))

    def compute_txy(self, processed):
        """
        Extract a numpy array of (time, x, y), discarding all other attributes
        """
        raise NotImplementedError

    def get_data(self, as_txy=True, **kwargs):
        res = self.load_raw_data(**kwargs)
        res = self.process_raw_data(res)
        self.set_idx_field(res)
        if self.convert_dates:
            self.date_conversion(res)
        if self.as_txy:
            res = self.compute_txy(res)
        return res, self.t0, self.idx



class DataLoaderDB(DataLoaderBase):

    model = None

    def __init__(self, *args, **kwargs):
        super(DataLoaderDB, self).__init__(*args, **kwargs)
        self.cursor = connection.cursor()

    def sql_get(self, **kwargs):
        raise NotImplementedError()

    def load_raw_data(self, **kwargs):
        sql = self.sql_get(**kwargs)
        self.cursor.execute(sql)
        return self.cursor.fetchall()


class DataLoaderFile(DataLoaderBase):

    def __init__(self, filename, fmt='pickle', *args, **kwargs):
        super(DataLoaderFile, self).__init__(*args, **kwargs)
        self.filename = filename
        self.fmt = fmt

    def load_raw_data(self, **kwargs):
        with open(self.filename, 'r') as f:
            if self.fmt.lower() == 'pickle':
                return pickle.load(f)
            if self.fmt.lower() == 'csv':
                c = csv.DictReader(f)
                return list(c)