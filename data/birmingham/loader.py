from ..loader import SpaceTimeFileLoader, DailyDataMixin, CsvFileMixin, PostgresqlDBMixin, SpaceTimeDatabaseLoader
try:
    from database import models
    NO_DB = False
except Exception:
    # disable loading from DB
    NO_DB = True
from database.consts import SRID
import consts
import datetime
import re
import numpy as np
from shapely import wkb, geometry
import fiona


class ResidentialBurglaryFileLoader(CsvFileMixin,
                                    DailyDataMixin,
                                    SpaceTimeFileLoader):

    index_key = 'crime_number'
    time_key = 'datetime_start'
    space_keys = ('x', 'y')

    @property
    def input_file(self):
        return consts.CRIME_DATA_FILE

    def parse_one(self, row):
        return {
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


class ResidentialBurglaryDBLoader(PostgresqlDBMixin,
                                  DailyDataMixin,
                                  SpaceTimeDatabaseLoader):
    table_name = 'birmingham'
    index_key = 'crime_number'
    time_key = 'datetime_start'
    space_keys = ('x', 'y')
    time_column = time_key
    space_column = 'location'
    srid = 27700

    @property
    def fields(self):
        return [
            ('crime_number',),
            ('offence',),
            ('datetime_start',),
            ('datetime_end',),
            ('ST_X(location)', 'x'),
            ('ST_Y(location)', 'y'),
            ('address',),
            ('officer_statement',),
        ]