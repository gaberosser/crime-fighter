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
    if isinstance(dt, datetime.date):
        if isinstance(dt, datetime.datetime):
            return dt
        else:
            return datetime.datetime.combine(dt, datetime.time(0))


def process_dt_end(dt):
    """
    Utility function that processes a date into a datetime, or does nothing if it already is one.
    Since this is for the END date, use the very end of the day.
    """
    oneday = datetime.timedelta(days=1)
    onems = datetime.timedelta(microseconds=1)
    if isinstance(dt, datetime.date):
        if isinstance(dt, datetime.datetime):
            return dt
        else:
            return datetime.datetime.combine(dt, datetime.time(0)) + oneday - onems


def start_of_the_day(dt):
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        the_date = dt
    else:
        the_date = dt.date()
    return datetime.datetime.combine(the_date, datetime.time(0))


class LoaderBase(object):

    def __init__(self, **kwargs):
        # set class attributes here
        self.errors = []
        super(LoaderBase, self).__init__(**kwargs)

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

    def raw_data(self):
        """
        :return: an iterable of raw data.
        This may be a generator (efficient for iterating through large files) or a list/array (e.g. in the case of
        a database query)
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

    def parse_raw_data(self):
        raw = self.raw_data()
        data = []
        for x in raw:
            if not self.pre_filter_one(x):
                continue
            try:
                t = self.parse_one(x)
            except Exception as exc:
                self.errors.append(exc)
                continue
            if self.post_filter_one(t):
                data.append(t)

    def get(self):
        parsed = self.parse_raw_data()
        return self.post_process(parsed)


# mixins
# file loaders

class FileLoaderMixin(object):
    @property
    def input_file(self):
        """
        Input file may be dependent upon the class arguments, for example if the data are split amongst multiple
        files.
        """
        raise NotImplementedError()


class CsvFileMixin(FileLoaderMixin):
    def raw_data(self):
        with open(self.input_file, 'rb') as f:
            c = csv.DictReader(f)
            for row in c:
                yield row


class PickleFileMixin(FileLoaderMixin):
    def raw_data(self):
        with open(self.input_file, 'rb') as f:
            raw_data = pickle.load(f)
        for r in raw_data:
            yield r

# database loaders

class DatabaseLoaderMixin(object):

    table_name = None

    def __init__(self, **kwargs):
        # set class attributes here
        self.cursor = None
        self.set_db_connection()
        super(DatabaseLoaderMixin, self).__init__(**kwargs)

    def set_db_connection(self):
        """
        See mixins for a variety of implementations
        """
        raise NotImplementedError

    @property
    def fields(self):
        """
        List of fields. Each element is EITHER a tuple of length one (the field name/expression) OR a tuple of length
        two (the field name/expression and the name to give it in the output)
        """
        return [
            ('col_name',),
            ('long_col_name', 'different_field_name'),
            ('FUNCTION(col_name)', 'modified_field_name')
        ]

    @property
    def sql_select(self):
        """
        Generate SQL code for the select portion of the query to extract data
        :return: (field string, substitution variables)
        """
        fields_str = []
        fields_var = []
        for x in self.fields:
            if len(x) == 2:
                fields_str.append("%s AS %s")
                fields_var.extend(x[:2])
            else:
                fields_str.append("%s")
                fields_var.append(x[0])
        fields_str = 'SELECT ' + ', '.join(fields_str)
        fields_str += '\n' +'FROM %s'
        fields_var.append(self.table_name)
        # substitution discouraged when it is user-supplied, but here it is all essentially hardcoded
        return fields_str % tuple(fields_var)

    @property
    def sql_where(self):
        """
        :return: (field string, substitution variables)
        """
        return "", []

    def raw_data(self):
        select_str = self.sql_select
        where_str, where_var = self.sql_where

        sql_str = '\n'.join([select_str, where_str])
        print sql_str

        self.cursor.execute(sql_str, where_var)
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
        Parse a single row of the raw data. The default behaviour is to use fields as a lookup.
        :param x:
        :return:
        """
        return dict([
            (k[1], x[i]) if len(k) == 2 else (k[0], x[i]) for i, k in enumerate(self.fields)
        ])


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

# time aggregators

class DailyDataMixin(object):
    def set_t0(self, data):
        if self.start_dt is not None:
            self.t0 = start_of_the_day(self.start_dt)
        else:
            t = [self.get_one_time(x) for x in data]
            self.t0 = start_of_the_day(min(t))

    def _convert_datetime(self, dt):
        return datetime_to_days(self.t0, dt)


class SpaceTimeLoader(LoaderBase):
    """
    time_key and space_keys refer to the PROCESSED KEYS that should be used to access the time and space data
    If one time_key is supplied, that is taken to be the precise datetime
    If two time keys are supplied, they are taken to be the start and end datetimes
    """
    time_key = None
    space_keys = (None,)
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
        self.start_dt = process_dt_start(start_dt)
        self.end_dt = process_dt_end(end_dt)
        self.domain = domain
        self.convert_dates = convert_dates
        self.to_txy = to_txy
        self.t0 = None
        if hasattr(self.time_key, '__iter__'):
            assert len(self.time_key) == 2, "Iterable time key only permissible for (start, end) formats"
            self.start_time_key = self.time_key[0]
            self.end_time_key = self.time_key[1]
        else:
            self.start_time_key = self.time_key
            self.end_time_key = self.time_key
        super(SpaceTimeLoader, self).__init__(**kwargs)


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
        for i, x in enumerate(data):
            this_t = self.get_one_time(x)
            if this_t is not None:
                val = self._convert_datetime(this_t)
            else:
                val = None
            self.set_one_time(x, val)

    def post_process(self, data):
        self.set_t0(data)
        if self.convert_dates:
            self.dates_to_float(data)

        # sort first by time, then by index to resolve equivalence
        data = sorted(data, key=lambda x: (self.get_one_time(x), x.get(self.index_key)))

        if self.to_txy:
            t = [self.get_one_time(r) for r in data]
            t = np.array(t).reshape((len(t), 1))
            space = [[r.get(ix) for ix in self.space_keys] for r in data]
            index = [r.get(self.index_key) for r in data]
            data = np.hstack((t, space))
        else:
            index = [r.get(self.index_key) for r in data]

        return data, index


class SpaceTimeFileLoader(FileLoaderMixin,
                          SpaceTimeLoader):
    def post_filter_one(self, x):
        filt = super(SpaceTimeFileLoader, self).post_filter_one(x)
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


class SpaceTimeDatabaseLoader(DatabaseLoaderMixin,
                              SpaceTimeLoader):

    """
    space_column is the name of the column holding the location Point geometry
    time_column is the name(s) of the column(s) holding the time data - either in (start, end) format or a single
    value
    srid is the SRID used to encode geo data
    """
    time_column = (None,)
    space_column = None
    srid = None

    def __init__(self, **kwargs):
        if hasattr(self.time_column, '__iter__'):
            assert len(self.time_column) == 2, "Iterable time col only permissible for (start, end) formats"
            self.start_time_column = self.time_column[0]
            self.end_time_column = self.time_column[1]
        else:
            self.start_time_column = self.time_column
            self.end_time_column = self.time_column
        super(SpaceTimeDatabaseLoader, self).__init__(**kwargs)

    @property
    def sql_where(self):
        """
        :return: (field string, substitution variables)
        """
        nstr = []
        nvar = []
        if self.start_dt is not None:
            nstr.append('{0} >= %s'.format(self.start_time_column))
            nvar.append(self.start_dt)
        if self.end_dt is not None:
            nstr.append('{0} <= %s'.format(self.end_time_column))
            nvar.append(self.end_dt)
        if self.domain is not None:
            nstr.append("{0} && ST_GeomFromText('%s', %s)".format(self.space_column))
            nvar.extend([self.domain.wkt, self.srid])
            nstr.append("ST_Intersects({0}, ST_GeomFromText('%s', %s))".format(self.space_column))
            nvar.extend([self.domain.wkt, self.srid])
        nstr = '\nAND '.join(nstr)

        pstr, pvar = super(SpaceTimeDatabaseLoader, self).sql_where
        if pstr == "" and nstr == "":
            return "", []

        where_str = "WHERE "
        where_var = []
        if pstr:
            where_str += pstr
            where_var += pvar
        if nstr:
            where_str += nstr
            where_var += nvar

            return where_str, where_var
        else:
            return "", []


class DataCombiner(object):
    def __init__(self, main_loader,
                 lookup_loader,
                 main_join_field,
                 lookup_join_field,
                 keep_null_join=True,
                 **kwargs):
        """
        Class for carrying out lookups when loading data.
        The lookup is performed on the parsed, pre-processed data.
        The joining field must be present in both
        The list of final output fields are provided as usual
        :param main_loader:
        :param lookup_loader:
        :param x_join_field: The name of the field in each case
        :param keep_null_join: If True, failed joins are maintained, but obv with no lookup data
        """
        self.main_loader = main_loader
        self.lookup_loader = lookup_loader
        self.main_join_field = main_join_field
        self.lookup_join_field = lookup_join_field
        self.main_data = self.main_loader.parse_raw_data()
        self.lookup_data = self.set_lookup_by_join()
        self.paired_data = None
        self.joined_data = None
        self.keep_null_join = keep_null_join

    @property
    def fields_main(self):
        """
        If a tuple of 1 then that is the name in the origin and final dataset
        If a tuple of 2, then the first is the original name and the second is the final name
        This is required if names clash in the two datasets.
        """
        return [
            ('the_name',),
            ('the_other_name', 'name_in_joined_data')]

    @property
    def fields_lookup(self):
        return []

    @staticmethod
    def main_join_to_lookup_join(mjoin):
        """
        Function that converts the main join column to the lookup join column
        """
        return mjoin

    def set_lookup_by_join(self):
        lookup_data = self.lookup_loader.parse_raw_data()
        lookup_by_join = {}
        for d in lookup_data:
            lookup_by_join[d.get(self.lookup_join_field)] = d
        return lookup_by_join

    def join_data(self):
        self.paired_data = []
        for x in self.main_data:
            idx = self.main_join_to_lookup_join(x.get(self.main_join_field))
            y = self.lookup_data.get(idx, None)
            if y is None and self.keep_null_join:
                self.paired_data.append((x, {}))
            else:
                self.paired_data.append((x, y))

    def combine_data(self):
        self.joined_data = []
        for x, y in self.paired_data:
            m = dict([
                (f[1 if len(f) == 2 else 0], x.get(f[0])) for f in self.fields_main
            ])
            l = dict([
                (f[1 if len(f) == 2 else 0], x.get(f[0])) for f in self.fields_lookup
            ])
            m.update(l)
            self.joined_data.append(m)

    def get(self):
        self.join_data()
        self.combine_data()
        return self.joined_data
