from ..network_loader import ITNLoader
from ..data_loader import DataLoaderFile, DataLoaderDB, datetime_to_days
try:
    from database import models
    NO_DB = False
except ImportError:
    # disable loading from DB
    NO_DB = True
from database.consts import SRID
import consts
import datetime
import re
import numpy as np
from shapely import wkb, geometry
import fiona


class BirminghamNetworkLoader(ITNLoader):
    srid = 27700
    net_dir = consts.NETWORK_DIR


def load_network():
    obj = BirminghamNetworkLoader('birmingham.gml')
    return obj.load()


def load_boundary_db(srid=SRID['uk']):
    obj = models.ArealUnit()
    if srid is not None:
        fields = ('ST_Transform(mpoly, {0})'.format(srid),)
    else:
        fields = ('mpoly',)
    res = obj.select(
        where_dict={'name': "'Birmingham'", 'type': "'city boundary'"},
        fields=fields,
        convert_to_dict=False)
    return wkb.loads(res[0][0], hex=True)


def load_boundary_file():
    ## TODO: doesn't support different projections at present
    fin = consts.BOUNDARY_SHAPEFILE
    with fiona.open(fin, 'r') as s:
        return geometry.shape(s[0]['geometry'])  # geojson object


class BirminghamCrimeFileLoader(DataLoaderFile):
    idx_field = 'crime_number'

    def process_raw_data(self, raw,
                         start_date=None,
                         end_date=None):
        if start_date and not isinstance(start_date, datetime.datetime):
            start_date = datetime.datetime.combine(start_date, datetime.time())
        if end_date and not isinstance(end_date, datetime.datetime):
            end_date = end_date + datetime.timedelta(days=1)
            end_date = datetime.datetime.combine(end_date, datetime.time()) - datetime.timedelta(seconds=1)
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
            if start_date and t['datetime_start'] < start_date:
                continue
            if end_date and t['datetime_start'] > end_date:
                continue
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

    def __init__(self, aoristic_method='start', max_time_window=None, *args, **kwargs):
        """
        :param max_time_window: MAximum time elapsed between start and end datetime in HOURS.
        """
        SUPPORTED_METHODS = (
            'start',
            'end',
            'mid'
        )
        if aoristic_method.lower() not in SUPPORTED_METHODS:
            raise ValueError("Unrecognised aoristic method. Supported values are %s" % ', '.join(SUPPORTED_METHODS))
        self.aoristic_method = aoristic_method
        self.max_time_window = max_time_window
        super(BirminghamCrimeLoader, self).__init__(*args, **kwargs)

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
        if self.max_time_window is not None:
            s = "EXTRACT(EPOCH FROM datetime_end - datetime_start) / 3600.0"
            where_dict[s] = "*<= {0}".format(self.max_time_window)

        fields = ('crime_number', 'datetime_start', 'datetime_end', 'ST_X(location)', 'ST_Y(location)')
        return obj.select(where_dict=where_dict or None, fields=fields, order_by='datetime_start')

    def process_raw_data(self, raw, **kwargs):
        res = []
        for row in raw:
            res.append({
                'crime_number': row['crime_number'],
                'datetime_start': row['datetime_start'],
                'datetime_end': row['datetime_end'],
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
            end = datetime_to_days(self.t0, t.pop('datetime_end'))
            t['t_end'] = end

    def compute_txy(self, processed):
        res = []
        for t in processed:
            if self.aoristic_method == 'start':
                if self.convert_dates:
                    res.append([t['t_start'], t['x'], t['y']])
                else:
                    res.append([t['datetime_start'], t['x'], t['y']])
            elif self.aoristic_method == 'end':
                if self.convert_dates:
                    res.append([t['t_end'], t['x'], t['y']])
                else:
                    res.append([t['datetime_end'], t['x'], t['y']])
            elif self.aoristic_method == 'mid':
                if self.convert_dates:
                    this_t = (t['t_end'] + t['t_start']) / 2.
                    res.append([this_t, t['x'], t['y']])
                else:
                    this_dt = (t['datetime_end'] - t['datetime_start']).total_seconds() / 2.
                    this_t = t['datetime_start'] + datetime.timedelta(seconds=this_dt)
                    res.append([this_t, t['x'], t['y']])
        return np.array(res)
