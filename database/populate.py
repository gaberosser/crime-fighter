__author__ = 'gabriel'
import os
import csv
import collections
import settings
import datetime
import time
import pytz
import models
import numpy as np
from django.contrib.gis.geos import Point, Polygon, MultiPolygon
from django.contrib.gis.utils import LayerMapping
from django.db import transaction
from django.db.models.signals import pre_save

UK_TZ = pytz.timezone('Europe/London')
CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')
CRIS_DATA_DIR = os.path.join(settings.DATA_DIR, 'cris')


def setup_ocu(**kwargs):
    OCU_CSV = os.path.join(CAD_DATA_DIR, 'ocu.csv')
    with open(OCU_CSV, 'r') as f:
        c = csv.reader(f)
        for row in c:
            ocu =  models.Ocu(code=row[0], description=row[1])
            ocu.save()


def setup_nicl(**kwargs):
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


def setup_cad(**kwargs):
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


def setup_cris(**kwargs):
    # precompile lists for speedier execution
    all_lsoa = dict([(x.code, x) for x in models.Division.objects.filter(type='lsoa')])
    all_nicl = dict([(x.number, x) for x in models.Nicl.objects.all()])

    CRIS_CSV = os.path.join(CRIS_DATA_DIR, 'cris-05_2010-04_2012.csv')
    with open(CRIS_CSV, 'r') as f:
        c = csv.reader(f)
        fields = c.next()
        bulk_list = []
        for row in c:
            res = {
                'lsoa_code': row[0],
                'date': datetime.date(int(row[1][0:4]), int(row[1][4:]), 1),
                'crime_major': row[2],
                'crime_minor': row[3],
                'count': int(row[4]),
            }
            cris = models.Cris(**res)
            cris.lsoa = all_lsoa.get(cris.lsoa_code)
            bulk_list.append(cris)
        print "Writing to database..."
        # write in batches to avoid massive memory requirements
        models.Cris.objects.bulk_create(bulk_list, batch_size=50000)
        print "Done"


def setup_boroughs(verbose=True):
    dt = models.DivisionType.objects.get(name='borough')

    def pre_save_callback(sender, instance, *args, **kwargs):
        instance.type = dt

    boroughs_mapping = {
        'name': 'NAME',
        'code': 'GSS_CODE',
        'mpoly': 'MULTIPOLYGON',
    }

    shp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'opendata/borough/London_Borough_Excluding_MHW.shp'))
    lm = LayerMapping(models.Division, shp_file, boroughs_mapping, transform=False)
    pre_save.connect(pre_save_callback, sender=models.Division)
    try:
        lm.save(strict=True, verbose=verbose)
    except Exception as exc:
        print repr(exc)
        raise
    finally:
        pre_save.disconnect(pre_save_callback, sender=models.Division)


def setup_lsoa_boundaries(verbose=True):
    dt = models.DivisionType.objects.get(name='lsoa')

    def pre_save_callback(sender, instance, *args, **kwargs):
        instance.type = dt

    mapping = {
        'name': 'LSOA11NM',
        'code': 'LSOA11CD',
        'mpoly': 'MULTIPOLYGON',
    }

    shp_file = os.path.join(settings.DATA_DIR, 'lsoa_boundaries_full_clipped/london/', 'london_only.shp')
    lm = LayerMapping(models.Division, shp_file, mapping, transform=False)
    pre_save.connect(pre_save_callback, sender=models.Division)
    try:
        lm.save(strict=True, verbose=verbose)
    except Exception as exc:
        print repr(exc)
        raise
    finally:
        pre_save.disconnect(pre_save_callback, sender=models.Division)


def setup_msoa_boundaries(verbose=True):
    dt = models.DivisionType.objects.get(name='msoa')

    def pre_save_callback(sender, instance, *args, **kwargs):
        instance.type = dt

    mapping = {
        'name': 'MSOA11NM',
        'code': 'MSOA11CD',
        'mpoly': 'MULTIPOLYGON',
    }

    shp_file = os.path.join(settings.DATA_DIR, 'msoa_boundaries_full_clipped/london/', 'london_only.shp')
    lm = LayerMapping(models.Division, shp_file, mapping, transform=False)
    pre_save.connect(pre_save_callback, sender=models.Division)
    try:
        lm.save(strict=True, verbose=verbose)
    except Exception as exc:
        print repr(exc)
        raise
    finally:
        pre_save.disconnect(pre_save_callback, sender=models.Division)


def setup_ward_boundaries(verbose=True):
    dt = models.DivisionType.objects.get(name='ward')

    def pre_save_callback(sender, instance, *args, **kwargs):
        instance.type = dt

    mapping = {
        'name': 'WD11NM',
        'code': 'WD11CD',
        'mpoly': 'MULTIPOLYGON',
    }

    shp_file = os.path.join(settings.DATA_DIR, 'ward_boundaries_full_extent/london/', 'london_only.shp')
    lm = LayerMapping(models.Division, shp_file, mapping, transform=False)
    pre_save.connect(pre_save_callback, sender=models.Division)
    try:
        lm.save(strict=True, verbose=verbose)
    except Exception as exc:
        print repr(exc)
        raise
    finally:
        pre_save.disconnect(pre_save_callback, sender=models.Division)


def setup_cad250_grid(verbose=True):
    import pdb
    dt = models.DivisionType.objects.get(name='cad_250m_grid')
    camden_borough = models.Division.objects.get(type='borough', name='Camden')
    x_coords = np.arange(523750, 532000, 250)
    y_coords = np.arange(180750, 188000, 250)
    I, J = np.meshgrid(range(len(x_coords) - 1), range(len(y_coords) - 1))

    polys = []

    for (i, j) in zip(I.flatten(), J.flatten()):
        poly = Polygon((
            (x_coords[i], y_coords[j]),
            (x_coords[i], y_coords[j+1]),
            (x_coords[i+1], y_coords[j+1]),
            (x_coords[i+1], y_coords[j]),
            (x_coords[i], y_coords[j]),
        ))
        if poly.distance(camden_borough.mpoly) < 1.0:
            polys.append(poly)

    divs = []

    for i, p in enumerate(polys):
        d = models.Division(name=str(i), code=str(i), type=dt, mpoly=MultiPolygon([p]))
        divs.append(d)

    try:
        models.Division.objects.bulk_create(divs)
    except Exception as exc:
        print repr(exc)
        raise exc
    else:
        if verbose:
            print "Created %u grid areas" % len(polys)


def setup_divisiontypes(**kwargs):
    # manually insert known division types
    dt, created = models.DivisionType.objects.get_or_create(name='borough')
    dt.description = 'London boroughs (also known as local authority)'
    dt.save()

    dt, created = models.DivisionType.objects.get_or_create(name='lsoa')
    dt.description = 'Lower super output area, combination of OAs'
    dt.save()

    dt, created = models.DivisionType.objects.get_or_create(name='msoa')
    dt.description = 'Middle super output area, combination of LSOAs'
    dt.save()

    dt, created = models.DivisionType.objects.get_or_create(name='ward')
    dt.description = 'Administrative unit for MPS, attached to a borough'
    dt.save()

    dt, created = models.DivisionType.objects.get_or_create(name='cad_250m_grid')
    dt.description = 'CAD 250m square grid'
    dt.save()


def setup_all(verbose=False):
    make_list = collections.OrderedDict([
        ('DIVISIONTYPE', setup_divisiontypes),
        ('OCU', setup_ocu),
        ('NICL', setup_nicl),
        ('BOROUGHS', setup_boroughs),
        ('WARDS', setup_ward_boundaries),
        ('LSOA', setup_lsoa_boundaries),
        ('MSOA', setup_msoa_boundaries),
        ('CADGRID', setup_cad250_grid),
        ('CAD', setup_cad),
        ('CRIS', setup_cris),
    ])
    for name, func in make_list.items():
        try:
            print "Populating %s data..." % name
            func(verbose=verbose)
            print "Success."
        except Exception as exc:
            print "Failed"
            print repr(exc)
    print "Completed all tasks"