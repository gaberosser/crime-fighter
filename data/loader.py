__author__ = 'gabriel'
import os
import csv
import pickle
import dill
import numpy as np
import datetime
from shapely import geometry


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
class DjangoDBMixin(object):
    def set_db_connection(self):
        from django.db import connection
        self.cursor = connection.cursor().cursor


class PostgresqlDBMixin(object):
    def set_db_connection(self):
        import psycopg2
        from settings import PSYCOPG2_DB_SETTINGS
        conn = psycopg2.connect(**PSYCOPG2_DB_SETTINGS)
        self.cursor = conn.cursor()


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
    # TODO: no references here to the base loader, so make this more like a MIXIN: remove post_filter_one then it can
    # be applied generically to DB *and* file loading
    """
    If one time_key is supplied, that is taken to be the precise datetime
    If two time keys are supplied, they are taken to be the start and end datetimes
    """
    time_key = None
    space_keys = (None, )
    index_key = None

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
            t = [self.get_one_time(r) for r in data]
            t = np.array(t).reshape((len(t), 1))
            space = [[r.get(ix) for ix in self.space_keys] for r in data]
            index = [r.get(self.index_key) for r in data]
            data = np.hstack((t, space))
        else:
            index = [r.get(self.index_key) for r in data]

        return data, index


class DatabaseLoader(object):

    table_name = None

    def __init__(self, **kwargs):
        # set class attributes here
        self.errors = []
        self.cursor = None
        self.set_db_connection()

    def set_db_connection(self):
        """
        See mixins for a variety of implementations
        """
        raise NotImplementedError

    @property
    def fields(self):
        return [
            ('col_name',),
            ('long_col_name', 'different_field_name'),
            ('FUNCTION(col_name)', 'modified_field_name')
        ]

    @property
    def sql_initial(self):
        """
        Generate SQL code for the first query to extract data
        """
        field_list = [" {0} AS {1}".format(x[0], x[1]) if len(x) == 2 else " {0}".format(x[0]) for x in self.fields]
        field_list = ','.join(field_list)
        sql = """SELECT {0} FROM {1}""".format(field_list,
                                               self.table_name)
        return sql

    def get_raw_data(self):
        self.cursor.execute(self.sql_initial)
        return self.cursor.fetchall()

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
        raw = self.get_raw_data()
        data = []
        for x in raw:
            try:
                t = self.parse_one(x)
            except Exception as exc:
                self.errors.append(exc)
                continue

            if self.post_filter_one(t):
                data.append(t)

        return self.post_process(data)
