__author__ = 'gabriel'
import os
import csv
import collections
import settings
import datetime
import time
import pytz
import models
from django.contrib.gis.geos import Point
from django.contrib.gis.utils import LayerMapping
from django.db import transaction

UK_TZ = pytz.timezone('Europe/London')
CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')


def setup_ocu():
    OCU_CSV = os.path.join(CAD_DATA_DIR, 'ocu.csv')
    with open(OCU_CSV, 'r') as f:
        c = csv.reader(f)
        for row in c:
            ocu =  models.Ocu(code=row[0], description=row[1])
            ocu.save()


def setup_nicl():
    NICL_CATEGORIES_CSV = os.path.join(CAD_DATA_DIR, 'nicl_categories.csv')
    with open(NICL_CATEGORIES_CSV, 'r') as f:
        c = csv.reader(f)
        for row in c:
            data = {
                'number': int(row[0]),
                'group': row[1],
                'description': row[2],
                'category_number': int(row[3]) if row[3] else None,
                'category_letter': row[4],
            }
            nicl = models.Nicl(**data)
            nicl.save()


def get_datetime(date_str, time_str):
    try:
        date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
        t = time.strptime(time_str, '%H:%M:%S')
    except ValueError:
        return None
    t = datetime.datetime.fromtimestamp(time.mktime(t)).time()
    dt = datetime.datetime.combine(date, t)
    return UK_TZ.localize(dt)


def get_ocu(code):
    try:
        return models.Ocu.objects.get(code=code)
    except (models.Ocu.MultipleObjectsReturned, models.Ocu.DoesNotExist):
        return None


def get_nicl(x):
    try:
        return models.Nicl.objects.get(pk=int(x))
    except Exception:
        return None


def get_point(x, y, srid):
    if x and y:
        return Point(int(x), int(y), srid=srid)
    else:
        return None


def parse_cad_rows(row, all_nicl, all_ocu):

        row = [x.strip() for x in row]
        return collections.OrderedDict(
            [
                ('call_sign', row[0] or None),
                ('res_type', row[1] or None),
                ('res_own', all_ocu.get(row[2])),
                ('inc_number', int(row[3]) if row[3] else None),
                ('inc_datetime', get_datetime(row[4], row[6])),
                ('inc_weekday', row[5] or None),
                ('op01', all_nicl.get(int(row[7] or -1))),
                ('op02', all_nicl.get(int(row[8] or -1))),
                ('op03', all_nicl.get(int(row[9] or -1))),
                ('att_bocu', all_ocu.get(row[10])),
                ('caller', row[17] or None),
                ('cl01', all_nicl.get(int(row[18] or -1))),
                ('cl02', all_nicl.get(int(row[19] or -1))),
                ('cl03', all_nicl.get(int(row[20] or -1))),
                ('cris_entry', row[21] or None),
                ('units_assigned_number', int(row[22]) if row[22] else 0),
                ('grade', row[23] or None),
                ('uc', row[24] == 'Y'),
                ('arrival_datetime', get_datetime(row[25], row[26])),
                ('response_time', row[28] or None),
                ('att_map', get_point(row[11], row[12], 27700)),
                ('inc_map', get_point(row[13], row[14], 27700)),
                ('call_map', get_point(row[15], row[16], 27700)),
                ]
        )


def setup_cad():
    # precompile lists for speedier execution
    all_nicl = dict([(x.number, x) for x in models.Nicl.objects.all()])
    all_ocu = dict([(x.code, x) for x in models.Ocu.objects.all()])

    CAD_CSV = os.path.join(CAD_DATA_DIR, 'mar2011-mar2012.csv')
    with open(CAD_CSV, 'r') as f:
        c = csv.reader(f)
        fields = c.next()
        # with transaction.atomic():
        bulk_list = []
        for row in c:
            res = parse_cad_rows(row, all_nicl, all_ocu)
            cad = models.Cad(**res)
            bulk_list.append(cad)
        print "Writing to database..."
        models.Cad.objects.bulk_create(bulk_list)
        print "Done"
            # try:
            #     cad.save()
            # except Exception as exc:
            #     print repr(exc)


def setup_boroughs(verbose=True):
    boroughs_mapping = {
        'name': 'NAME',
        'area': 'HECTARES',
        'nonld_area': 'NONLD_AREA',
        'mpoly': 'MULTIPOLYGON',
    }

    shp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'opendata/borough/London_Borough_Excluding_MHW.shp'))
    lm = LayerMapping(models.Borough, shp_file, boroughs_mapping, transform=False)
    try:
        lm.save(strict=True, verbose=verbose)
    except Exception as exc:
        print repr(exc)
        raise
