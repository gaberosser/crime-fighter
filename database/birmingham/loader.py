from ..network_loader import ITNLoader
from ..data_loader import DataLoaderFile, DataLoaderDB, datetime_to_days
try:
    from database import models
    NO_DB = False
except ImportError:
    # disable loading from DB
    NO_DB = True
import consts
import datetime
import re
import numpy as np


class BirminghamNetworkLoader(ITNLoader):
    srid = 27700
    net_dir = consts.NETWORK_DIR


def load_network():
    obj = BirminghamNetworkLoader('birmingham.gml')
    return obj.load()


class BirminghamCrimeFileLoader(DataLoaderFile):
    idx_field = 'crime_number'

    def process_raw_data(self, raw, **kwargs):
        res = []
        for row in raw:
            t = {
                'crime_number': row['CRIME_NO'],
                'offence': row['OFFENCE'],
                'datetime_start': datetime.datetime.strptime('%s-%d:%d' % (
                    row['DATE_START'],
                    int(row['HR_START']),
                    int(row['MIN_START'])), '%d/%m/%Y-%H:%M'),
                'datetime_end': datetime.datetime.strptime('%s-%d:%d' % (
                    row['DATE_END'],
                    int(row['HR_END']),
                    int(row['MIN_END'])), '%d/%m/%Y-%H:%M'),
                'x': int(row['EASTING']),
                'y': int(row['NORTHING']),
                'address': row['FULL_LOC'].replace("'", ""),
                'officer_statement': re.sub(r'[^\x00-\x7f]', r'', row['MO']).replace("'", "")
            }
            res.append(t)
        return res

    def date_conversion(self, processed):
        min_start = min([t['datetime_start'] for t in processed])
        self.t0 = min_start
        for t in processed:
            start = datetime_to_days(self.t0, t.pop('datetime_start'))
            end = datetime_to_days(self.t0, t.pop('datetime_end'))
            t['t_start'] = start
            t['t_end'] = end

    def compute_txy(self, processed):
        res = []
        for t in processed:
            if self.convert_dates:
                res.append([t['t_start'], t['x'], t['y']])
            else:
                res.append([t['datetime_start'], t['x'], t['y']])
        return np.array(res)


class BirminghamCrimeLoader(DataLoaderDB):
    idx_field = 'crime_number'

    @property
    def model(self):
        return models.Birmingham

    def load_raw_data(self,
                      crime_type=None,
                      start_date=None,
                      end_date=None,
                      domain=None):
        """
        Get all matching crimes from the Chicago dataset
        :param crime_type:
        :param start_date:
        :param end_date:
        :param domain: shapely object
        :param convert_dates: If True, dates are converted to the number of days after t0 (float)
        :param where_strs:
        :return:
        """
        if start_date and not isinstance(start_date, datetime.datetime):
            # start_date refers to 00:00 onwards
            start_date = datetime.datetime.combine(start_date, datetime.time(0))
        if end_date and not isinstance(end_date, datetime.datetime):
            # end_date refers to 23:59:59 backwards
            end_date = end_date + datetime.timedelta(days=1)
            end_date = datetime.datetime.combine(end_date, datetime.time(0)) - datetime.timedelta(seconds=1)

        obj = self.model()

        where_dict = {}

        if crime_type:
            where_dict['LOWER(type)'] = "*LIKE '{0}'".format(crime_type.lower())
        if start_date:
            # where_dict['datetime_end'] = "*>= '{0}'".format(start_date.strftime('%Y-%m-%d %H:%M:%S'))
            where_dict['datetime_start'] = "*>= '{0}'".format(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        if end_date:
            where_dict['"datetime_start"'] = "*<= '{0}'".format(end_date.strftime('%Y-%m-%d %H:%M:%S'))
        if domain:
            s = "ST_Intersects(location, ST_GeomFromText('{0}', {1}))".format(domain.wkt, obj.srid)
            where_dict[s] = '*'
        fields = ('crime_number', 'datetime_start', 'ST_X(location)', 'ST_Y(location)')
        return obj.select(where_dict=where_dict or None, fields=fields, order_by='datetime_start')

    def process_raw_data(self, raw):
        res = []
        for row in raw:
            res.append({
                'crime_number': row['crime_number'],
                'datetime_start': row['datetime_start'],
                'x': row['ST_X(location)'],
                'y': row['ST_Y(location)'],
            })
        return res

    def date_conversion(self, processed):
        min_start = min([t['datetime_start'] for t in processed])
        self.t0 = min_start
        for t in processed:
            start = datetime_to_days(self.t0, t.pop('datetime_start'))
            t['t_start'] = start

    def compute_txy(self, processed):
        res = []
        for t in processed:
            if self.convert_dates:
                res.append([t['t_start'], t['x'], t['y']])
            else:
                res.append([t['datetime_start'], t['x'], t['y']])
        return np.array(res)
