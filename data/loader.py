__author__ = 'gabriel'
import os
import csv
import pickle
import dill
import numpy as np
import datetime
from shapely import geometry
try:
    from django.db import connection
    NO_DB = False
except ImportError:
    # disable loading from DB
    NO_DB = True


def datetime_to_days(t0, dt):
    delta = dt - t0
    return delta.total_seconds() / 86400.


def process_dt_start(dt):
    """
    Utility function that processes a date into a datetime, or does nothing if it already is one.
    Since this is for the START date, use the very beginning of the day.
    """
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        return datetime.datetime.combine(dt, datetime.time(0))


def process_dt_end(dt):
    """
    Utility function that processes a date into a datetime, or does nothing if it already is one.
    Since this is for the END date, use the very end of the day.
    """
    oneday = datetime.timedelta(days=1)
    onems = datetime.timedelta(microseconds=1)
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        return datetime.datetime.combine(dt, datetime.time(0)) + oneday - onems


def start_of_the_day(dt):
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        the_date = dt
    else:
        the_date = dt.date()
    return datetime.datetime.combine(the_date, datetime.time(0))


class FileLoader(object):

    def __init__(self, **kwargs):
        # set class attributes here
        self.errors = []

    @property
    def input_file(self):
        """
        Input file may be dependent upon the class arguments, for example if the data are split amongst multiple
        files.
        """
        raise NotImplementedError()

    def pre_filter_one(self, rawrow):
        """
        Apply filter to a RAW datum
        :param x:
        :return: True if the row passes, otherwise False
        """
        return True

    def post_filter_one(self, parsedrow):
        """
        Apply filter to a PARSED datum
        :param x:
        :return: True if the row passes, otherwise False
        """
        return True

    def parse_one(self, x):
        """
        Parse a single row of the raw data
        :param x:
        :return:
        """
        return x

    def post_process(self, data):
        """
        Optionally process the parsed data block before it is returned
        """
        return data

    def raw_generator(self):
        """
        :return: a generator of raw data.
        """
        raise NotImplementedError()

    def save(self, outfile, overwrite=False, method='pickle'):
        if os.path.exists(outfile) and not overwrite:
            raise AttributeError("%s already exists and overwrite=False", outfile)
        data = self.get()
        with open(outfile, 'wb') as f:
            if method == 'pickle':
                pickle.dump(data, f)
            elif method == 'dill':
                dill.dump(data, f)

    def dill(self, outfile, overwrite=False):
        self.save(outfile, overwrite=overwrite, method='dill')

    def get(self):
        gen = self.raw_generator()
        data = []
        for x in gen:
            if not self.pre_filter_one(x):
                continue

            try:
                t = self.parse_one(x)
            except Exception as exc:
                self.errors.append(exc)
                continue

            if self.post_filter_one(t):
                data.append(t)

        return self.post_process(data)


# mixins

class CsvFileMixin(object):
    def raw_generator(self):
        with open(self.input_file, 'rb') as f:
            c = csv.DictReader(f)
            for row in c:
                yield row


class PickleFileMixin(object):
    def raw_generator(self):
        with open(self.input_file, 'rb') as f:
            raw_data = pickle.load(f)
        for r in raw_data:
            yield r


class DailyDataMixin(object):
    def set_t0(self, data):
        if self.start_dt is not None:
            self.t0 = start_of_the_day(self.start_dt)
        else:
            t = [self.get_one_time(x) for x in data]
            self.t0 = start_of_the_day(min(t))

    def _convert_datetime(self, dt):
        return datetime_to_days(self.t0, dt)


class STFileLoader(FileLoader):

    """
    If one time_key is supplied, that is taken to be the precise datetime
    If two time keys are supplied, they are taken to be the start and end datetimes
    """
    time_key = None
    space_keys = (None, )

    def __init__(self,
                 start_dt=None,
                 end_dt=None,
                 domain=None,
                 convert_dates=True,
                 to_txy=True,
                 **kwargs):
        """
        :param start_dt:
        :param end_dt:
        :param domain:
        :param convert_dates: If True, dates are converted to floats, relative to the beginning of the day on start_dt
        if present, otherwise the start of the first date in the data.
        :param to_txy: If True, only the time and space components of the data are returned
        :param kwargs:
        :return:
        """
        self.start_dt = None
        if start_dt is not None:
            self.start_dt = process_dt_start(start_dt)
        self.end_dt = None
        if end_dt is not None:
            self.end_dt = process_dt_end(end_dt)
        self.domain = domain
        self.convert_dates = convert_dates
        self.to_txy = to_txy
        self.t0 = None
        super(STFileLoader, self).__init__(**kwargs)

    def get_one_time(self, x):
        """
        Override this behaviour with a mixin if a midpoint between start and end is required
        :param x:
        :return:
        """
        return x.get(self.time_key)

    def set_one_time(self, x, val):
        """
        Override this behaviour with a mixin if a midpoint between start and end is required
        """
        x[self.time_key] = val

    def dates_to_float(self, data):
        """
        Modify data in-place, converting datetimes to floats
        self.t0 has already been set by this stage
        """
        for x in data:
            this_t = self.get_one_time(x)
            if this_t is not None:
                val = self._convert_datetime(this_t)
            self.set_one_time(x, val)

    def post_filter_one(self, x):
        filt = super(STFileLoader, self).post_filter_one(x)
        if not filt:
            return False
        if self.start_dt is not None:
            if not (x.get(self.time_key) >= self.start_dt):
                return False
        if self.end_dt is not None:
            if not (x.get(self.time_key) <= self.end_dt):
                return False
        if self.domain is not None:
            pt = geometry.Point(*[x.get(ix) for ix in self.space_keys])
            if not pt.within(self.domain):
                return False
        return True

    def post_process(self, data):
        self.set_t0(data)
        if self.convert_dates:
            self.dates_to_float(data)
            data = sorted(data, key=lambda x: self.get_one_time(x))
        if self.to_txy:
            t = [r.get(self.time_key) for r in data]
            t = np.array(t).reshape((len(t), 1))
            space = [[r.get(ix) for ix in self.space_keys] for r in data]
            data = np.hstack((t, space))

        return super(STFileLoader, self).post_process(data)


class DataLoaderBase(object):

    idx_field = None

    def __init__(self, convert_dates=True, as_txy=True):
        self.convert_dates = convert_dates
        self.as_txy = as_txy
        self.t0 = None
        self.idx = None

    def load_raw_data(self, **kwargs):
        raise NotImplementedError()

    def process_raw_data(self, raw, **kwargs):
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
        # send the same kwargs to both
        # this only works because at present EITHER one OR the other method uses them
        res = self.load_raw_data(**kwargs)
        res = self.process_raw_data(res, **kwargs)
        self.set_idx_field(res)
        if self.convert_dates:
            self.date_conversion(res)
        if self.as_txy:
            res = self.compute_txy(res)
        return res, self.t0, self.idx


class DataLoaderDB(DataLoaderBase):

    @property
    def model(self):
        return None

    def __init__(self, *args, **kwargs):
        if NO_DB:
            raise NotImplementedError("No DB support on this system")
        else:
            super(DataLoaderDB, self).__init__(*args, **kwargs)
            self.cursor = connection.cursor()

    def sql_get(self, **kwargs):
        # filtering happens here using kwargs
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

    def process_raw_data(self, raw, **kwargs):
        # filtering happens here using kwargs
        return super(DataLoaderFile, self).process_raw_data(raw, **kwargs)