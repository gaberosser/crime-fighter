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
from django.contrib.gis.gdal import DataSource
from django.db import transaction
from django.db.models.signals import pre_save
from django.db import connection
import re
import shapefile
import fiona  # nicer way to interact with shapefiles

UK_TZ = pytz.timezone('Europe/London')
USCENTRAL_TZ = pytz.timezone('US/Central')
CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')
CRIS_DATA_DIR = os.path.join(settings.DATA_DIR, 'cris')
CHICAGO_DATA_DIR = os.path.join(settings.DATA_DIR, 'chicago')


def sql_quote(x):
    return "'%s'" % x


def setup_ocu(**kwargs):
    OCU_CSV = os.path.join(CAD_DATA_DIR, 'ocu.csv')
    with open(OCU_CSV, 'r') as f:
        c = csv.DictReader(f)
        for row in c:
            ocu =  models.Ocu(code=row['code'], description=row['interpretation'])
            ocu.save()


def setup_nicl(**kwargs):
    NICL_CATEGORIES_CSV = os.path.join(CAD_DATA_DIR, 'nicl_categories.csv')
    with open(NICL_CATEGORIES_CSV, 'r') as f:
        c = csv.reader(f)
        # read header
        fields = c.next()
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
    # delete existing boroughs
    models.Division.objects.filter(type=dt).delete()

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


def setup_cad250_grid(verbose=True, test=False):
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

    if test:
        try:
            [x.save() for x in divs]
        except Exception as exc:
            ## FIXME: I have NO idea why this needs to be called twice in test mode
            [x.save() for x in divs]

    else:
        try:
            models.Division.objects.bulk_create(divs)
        except Exception as exc:
            print repr(exc)
            raise exc

    if verbose:
        print "Created %u grid areas" % len(polys)

# INSERT into spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) values ( 102671, 'esri', 102671, '+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 +x_0=300000 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs ', 'PROJCS["NAD_1983_StatePlane_Illinois_East_FIPS_1201_Feet",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",984249.9999999999],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",-88.33333333333333],PARAMETER["Scale_Factor",0.999975],PARAMETER["Latitude_Of_Origin",36.66666666666666],UNIT["Foot_US",0.30480060960121924],AUTHORITY["EPSG","102671"]]');
def setup_chicago_community_area(**kwargs):
    dt = models.DivisionType.objects.get(name='chicago_community_area')
    shp_file = os.path.join(CHICAGO_DATA_DIR, 'community_areas', 'community_areas.shp')
    ds = DataSource(shp_file)
    lyr = ds[0]
    # source_srid = 102671
    # source_srid = 26971
    dest_srid = 2028

    for x in lyr:
        name = x.get('COMMUNITY')
        mpoly = x.geom
        mpoly.srid = dest_srid
        # mpoly.srid = source_srid
        # mpoly.transform(dest_srid)
        mpoly = mpoly.geos
        if isinstance(mpoly, Polygon):
            mpoly = MultiPolygon(mpoly)

        try:
            m = models.ChicagoDivision.objects.get(name=name, type=dt)
        except models.ChicagoDivision.DoesNotExist:
            m = models.ChicagoDivision(name=name, type=dt)
        m.mpoly = mpoly
        m.save()

    if kwargs.pop('verbose', False):
        print "Created Chicago community areas"


# not required - single mpoly is easy to load straight from file
# before running: need to make sure PostGIS has the weird SRS coord system used here
# INSERT into spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) values ( 102671, 'esri', 102671, '+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 +x_0=300000 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs ', 'PROJCS["NAD_1983_StatePlane_Illinois_East_FIPS_1201_Feet",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",984249.9999999999],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",-88.33333333333333],PARAMETER["Scale_Factor",0.999975],PARAMETER["Latitude_Of_Origin",36.66666666666666],UNIT["Foot_US",0.30480060960121924],AUTHORITY["EPSG","102671"]]');

def setup_chicago_division(**kwargs):
    dt = models.DivisionType.objects.get(name='city')
    shp_file = os.path.join(CHICAGO_DATA_DIR, 'community_areas', 'community_areas.shp')
    ds = DataSource(shp_file)
    lyr = ds[0]
    # source_srid = 102671
    # source_srid = 26971
    dest_srid = 2028

    mpolys = []

    for x in lyr:
        mpoly = x.geom
        mpoly.srid = dest_srid
        # mpoly.srid = source_srid
        # mpoly.transform(dest_srid)
        mpoly = mpoly.geos
        if isinstance(mpoly, Polygon):
            mpoly = MultiPolygon(mpoly)
        mpolys.append(mpoly)

    mpoly = reduce(lambda x, y: x.union(y), mpolys)

    try:
        m = models.ChicagoDivision.objects.get(name='Chicago', type=dt)
    except models.ChicagoDivision.DoesNotExist:
        m = models.ChicagoDivision(name='Chicago', type=dt)
    m.mpoly = mpoly
    m.save()

    # filled in version
    mls = mpoly.boundary
    x = mls[0].coords
    x += (x[0],)
    mpoly = MultiPolygon(Polygon(x))

    try:
        m = models.ChicagoDivision.objects.get(name='ChicagoFilled', type=dt)
    except models.ChicagoDivision.DoesNotExist:
        m = models.ChicagoDivision(name='ChicagoFilled', type=dt)
    m.mpoly = mpoly
    m.save()

    if kwargs.pop('verbose', False):
        print "Created Chicago city region"


def setup_chicago_sides(**kwargs):

    dt = models.DivisionType.objects.get(name='chicago_side')
    mappings = {
        'Central': [
            'Near North Side',
            'Loop',
            'Near South Side',
        ],
        'North': [
            'North Center',
            'Lake View',
            'Lincoln Park',
            'Avondale',
            'Logan Square'
        ],
        'Far North': [
            'Rogers Park',
            'West Ridge',
            'Uptown',
            'Lincoln Square',
            'Edison Park',
            'Norwood Park',
            'Jefferson Park',
            'Forest Glen',
            'North Park',
            'Albany Park',
            "OHare",
            'Edgewater',
        ],
        'Northwest': [
            'Portage Park',
            'Irving Park',
            'Dunning',
            'Montclare',
            'Belmont Cragin',
            'Hermosa',
        ],
        'West': [
            'Humboldt Park',
            'West Town',
            'Austin',
            'West Garfield Park',
            'East Garfield Park',
            'Near West Side',
            'North Lawndale',
            'South Lawndale',
            'Lower West Side'
        ],
        'South': [
            'Armour Square',
            'Douglas',
            'Oakland',
            'Fuller Park',
            'Grand Boulevard',
            'Kenwood',
            'Washington Park',
            'Hyde Park',
            'Woodlawn',
            'South Shore',
            'Bridgeport',
            'Greater Grand Crossing',
        ],
        'Southwest': [
            'Garfield Ridge',
            'Archer Heights',
            'Brighton Park',
            'McKinley Park',
            'New City',
            'West Elsdon',
            'Gage Park',
            'Clearing',
            'West Lawn',
            'Chicago Lawn',
            'West Englewood',
            'Englewood',
        ],
        'Far Southeast': [
            'Chatham',
            'Avalon Park',
            'South Chicago',
            'Burnside',
            'Calumet Heights',
            'Roseland',
            'Pullman',
            'South Deering',
            'East Side',
            'West Pullman',
            'Riverdale',
            'Hegewisch',
        ],
        'Far Southwest': [
            'Ashburn',
            'Auburn Gresham',
            'Beverly',
            'Washington Heights',
            'Mount Greenwood',
            'Morgan Park',
        ]
    }

    # not_found = collections.defaultdict(list)
    for k, v in mappings.items():
        cas = []
        # try to find all CAs
        for x in v:
            try:
                cas.append(models.ChicagoDivision.objects.get(type='chicago_community_area', name__iexact=x).mpoly)
            except Exception:
                # not_found[k].append(x)
                raise
        mpoly = reduce(lambda x, y: x.union(y), cas)
        try:
            m = models.ChicagoDivision.objects.get(type=dt, name=k)
        except models.ChicagoDivision.DoesNotExist:
            m = models.ChicagoDivision(type=dt, name=k)

        if isinstance(mpoly, Polygon):
            mpoly = MultiPolygon(mpoly)

        m.mpoly = mpoly
        m.save()

    if kwargs.pop('verbose', False):
        print "Created Chicago sides"


def setup_divisiontypes(**kwargs):
    res = []
    # manually insert known division types
    dt, created = models.DivisionType.objects.get_or_create(name='borough')
    dt.description = 'London boroughs (also known as local authority)'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='lsoa')
    dt.description = 'Lower super output area, combination of OAs'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='msoa')
    dt.description = 'Middle super output area, combination of LSOAs'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='ward')
    dt.description = 'Administrative unit for MPS, attached to a borough'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='cad_250m_grid')
    dt.description = 'CAD 250m square grid'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='monsuru_250m_grid')
    dt.description = 'Monsuru CAD 250m square grid'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='city')
    dt.description = 'City region'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='chicago_community_area')
    dt.description = 'Community areas in the City of Chicago'
    dt.save()
    res.append(dt.name)

    dt, created = models.DivisionType.objects.get_or_create(name='chicago_side')
    dt.description = 'Sides in the City of Chicago'
    dt.save()
    res.append(dt.name)

    if kwargs.pop('verbose', False):
        print "Saved %d records: %s" % (len(res), ",".join(res))


@transaction.commit_manually
def setup_chicago_data(verbose=True, chunksize=50000):
    def point_2028(lat, long):
        p = Point(long, lat, srid=4326)
        p.transform(2028)
        return p

    mappings = {
        'number': lambda x: int(x.get('ID')),
        'case_number': lambda x: x.get('Case Number'),
        'datetime': lambda x: datetime.datetime.strptime(x.get('Date'), '%m/%d/%Y %I:%M:%S %p').replace(
            tzinfo=USCENTRAL_TZ),
        'block': lambda x: x.get('Block'),
        'iucr': lambda x: x.get('IUCR'),
        'primary_type': lambda x: x.get('Primary Type'),
        'description': lambda x: x.get('Description'),
        'location_type': lambda x: x.get('Location Description'),
        'arrest': lambda x: x.get('Arrest') == 'true',
        'domestic': lambda x: x.get('Domestic') == 'true',
        'location': lambda x: point_2028(float(x.get('Latitude')), float(x.get('Longitude'))),
    }

    count = 0
    fail_count = 0

    with open(os.path.join(CHICAGO_DATA_DIR, 'chicago_crime_data.csv'), 'r') as f:
        c = csv.DictReader(f)
        try:
            for row in c:
                try:
                    t = models.Chicago(**dict([(x, y(row)) for x, y in mappings.items()]))
                except Exception:
                    fail_count += 1
                    continue
                t.save()
                count += 1
                if (count % chunksize) == 0 and count:
                    transaction.commit()
                    print count
        except Exception as exc:
            transaction.rollback()
            raise exc
        else:
            transaction.commit()

    if verbose:
        print "Saved %d Chicago crime records.  %d failed." % (count, fail_count)


def setup_monsuru_cad_grid(verbose=False):
    DATADIR = os.path.join(settings.DATA_DIR, 'monsuru/cad_grids')
    fin = os.path.join(DATADIR, 'grids')
    s = shapefile.Reader(fin)
    dt = models.DivisionType.objects.get(name='monsuru_250m_grid')

    count = 0

    for x in s.shapeRecords():
        poly = Polygon([tuple(t) for t in x.shape.points])
        div = models.Division(
            type=dt,
            name=str(x.record[5]),
            mpoly=MultiPolygon(poly),
        )
        div.save()
        count += 1

    if verbose:
        print "Saved %d grid records" % count


def setup_all(verbose=False):
    make_list = collections.OrderedDict([
        ('DIVISIONTYPE', setup_divisiontypes),
        ('OCU', setup_ocu),
        ('NICL', setup_nicl),
        ('BOROUGHS', setup_boroughs),
        ('WARDS', setup_ward_boundaries),
        # ('LSOA', setup_lsoa_boundaries),
        # ('MSOA', setup_msoa_boundaries),
        ('CADGRID', setup_cad250_grid),
        ('CAD', setup_cad),
        ('CRIS', setup_cris),
        ('CHICAGODIVISION', setup_chicago_division),
        ('CHICAGOSIDES', setup_chicago_sides),
        ('CHICAGOCOMMUNITYAREA', setup_chicago_community_area),
        ('CHICAGODATA', setup_chicago_data),
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


def import_monsuru_cad_data():

    DATADIR = os.path.join(settings.DATA_DIR, 'monsuru/cad')
    def date_parser(x):
        if hasattr(x, '__iter__'):
            return datetime.date(x[0], x[1], x[2])
        t0 = datetime.date(2011, 3, 1)
        return t0 + datetime.timedelta(days=int(x - 40603))

    cur = connection.cursor()
    input_dict = {
        'monsuru_cad_burglary': 'burglary',
        'monsuru_cad_violence': 'Violence',
        'monsuru_cad_shoplifting': 'Shoplifting',
    }
    create_sql = """CREATE TABLE {0} (id INTEGER PRIMARY KEY, inc_date DATE);
                    SELECT AddGeometryColumn('{0}', 'location', 27700, 'POINT', 2);"""
    pt_from_text_sql = """ST_GeomFromText('POINT(%f %f)', 27700)"""

    for t, v in input_dict.items():
        # DROP if possible
        try:
            cur.execute("DROP TABLE %s;" % t)
        except Exception as exc:
            print repr(exc)
            pass
        # CREATE
        print "Create table %s..." % t
        cur.execute(create_sql.format(t))
        print "Done."
        # POPULATE
        fin = os.path.join(DATADIR, v)
        s = shapefile.Reader(fin)
        sn1_idx = [zz[0].lower() for zz in s.fields].index('sn1') - 1  # first field entry doesn't actually show up
        print "Populating table %s..." % t
        for x in s.shapeRecords():
            # import ipdb; ipdb.set_trace()
            insert_sql = """INSERT INTO {0} (id, inc_date, location) VALUES ({1}, '{2}', {3});""".format(
                t,
                x.record[sn1_idx],
                date_parser(x.record[2]).strftime('%Y-%m-%d'),
                pt_from_text_sql % (x.shape.points[0][0], x.shape.points[0][1])
            )
            cur.execute(insert_sql)
        print "Done."


def populate_geos_table(table_obj, data, mapper, skip_clause=None, verbose=True, chunksize=50000, limit=None):
    table_obj.rewrite_table()
    count = 0
    res = []
    for row in data:
        if limit and count > limit:
            break
        if skip_clause is not None and skip_clause(row):
            continue
        try:
            datum = dict([(k, v(row)) for k, v in mapper.items()])
        except Exception as exc:
            import ipdb; ipdb.set_trace()
        res.append(datum)
        count += 1
        if (count % chunksize) == 0 and count:
            table_obj.insert_many(res)
            res = []
    table_obj.insert_many(res)
    return count


def chicago(verbose=True, chunksize=50000, limit=None):
    obj = models.Chic()
    DATADIR = os.path.join(settings.DATA_DIR, 'chicago')

    mapper = {
        'id': lambda x: int(x.get('ID')),
        'case_number': lambda x: sql_quote(x.get('Case Number')),
        'datetime': lambda x: sql_quote(datetime.datetime.strptime(x.get('Date'), '%m/%d/%Y %I:%M:%S %p').replace(
            tzinfo=USCENTRAL_TZ)),
        'primary_type': lambda x: sql_quote(x.get('Primary Type')),
        'description': lambda x: sql_quote(x.get('Description')),
        'arrest_made': lambda x: x.get('Arrest') == 'true',
        'location': lambda x: "ST_Transform(ST_SetSRID(ST_Point({0}, {1}), 4326), 2028)".format(
                    float(x['Longitude']),
                    float(x['Latitude'])
                ),
        # 'block': lambda x: x.get('Block'),
        # 'iucr': lambda x: x.get('IUCR'),
        # 'location_type': lambda x: x.get('Location Description'),
        # 'domestic': lambda x: x.get('Domestic') == 'true',
    }

    skip_clause = lambda x: x['Location'] == ''

    with open(os.path.join(CHICAGO_DATA_DIR, 'chicago_crime_data.csv'), 'r') as f:
        data = csv.DictReader(f)
        count = populate_geos_table(obj, data, mapper,
                                    skip_clause=skip_clause,
                                    verbose=verbose,
                                    chunksize=chunksize,
                                    limit=limit)
    print count



def san_francisco(verbose=True, chunksize=50000, limit=None):
    obj = models.SanFrancisco()
    obj.rewrite_table()

    DATADIR = os.path.join(settings.DATA_DIR, 'san_francisco')
    count = 1
    with open(os.path.join(DATADIR, 'san_fran_crime_from_1_jan_2003.csv'), 'r') as f:
        c = csv.DictReader(f)
        res = []
        for row in c:
            if limit and count > limit:
                break
            t = {
                'incident_number': row['IncidntNum'],
                'datetime': "to_timestamp('{0} {1}', 'MM/DD/YYYY HH24:MI')".format(
                    row['Date'].split(' ')[0],
                    row['Time']
                ),
                'location': "ST_Transform(ST_SetSRID(ST_Point({0}, {1}), 4326), {2})".format(
                    float(row['X']),
                    float(row['Y']),
                    models.SRID['san francisco']
                ),
                'category': "'%s'" % row['Category'],
            }
            res.append(t)
            count += 1
            if (count % chunksize) == 0 and count:
                obj.insert_many(res)
                res = []
    obj.insert_many(res)
    if verbose:
        print "Saved %d crime records" % count


def san_francisco_boundary():
    import json
    DATADIR = os.path.join(settings.DATA_DIR, 'san_francisco')
    fin = os.path.join(DATADIR, 'boundary', 'san_francisco_boundary.shp')
    obj = models.SanFranciscoDivision()
    with fiona.open(fin, 'r') as s:
        res = s[0]['geometry']  # geojson format
        insert_sql = {
            'name': sql_quote('San Francisco'),
            'type': sql_quote('city boundary'),
            'mpoly': "ST_SetSRID(ST_GeomFromGeoJSON('%s'), %d)" % (json.dumps(res), models.SRID['san francisco']),
        }
        obj.insert(**insert_sql)


def los_angeles(verbose=True, chunksize=50000, limit=None):
    obj = models.LosAngeles()
    obj.rewrite_table()
    srid = models.SRID['los angeles']

    DATADIR = os.path.join(settings.DATA_DIR, 'los_angeles')
    count = 1
    with open(os.path.join(DATADIR, 'lapd_crime_and_collision_2014.csv'), 'r') as f:
        c = csv.DictReader(f)
        res = []
        for row in c:
            if limit and count > limit:
                break
            if len(row['Location 1']) == 0 or row['Location 1'][0] != '(':
                continue
            loc_comp = row['Location 1'].replace(' ', '')[1:-1].split(',')
            lat = float(loc_comp[0])
            lon = float(loc_comp[1])

            t = {
                'incident_number': row['DR NO'],
                'datetime': "to_timestamp('{0} {1}', 'MM/DD/YYYY HH24MI')".format(
                    row['DATE OCC'],
                    '%04d' % int(row['TIME OCC'])
                ),
                'location': "ST_Transform(ST_SetSRID(ST_Point({0}, {1}), 4326), {2})".format(
                    lon,
                    lat,
                    srid
                ),
                'category': "'%s'" % row['Crm Cd Desc'],
            }
            res.append(t)
            count += 1
            if (count % chunksize) == 0 and count:
                obj.insert_many(res)
                res = []
    obj.insert_many(res)
    if verbose:
        print "Inserted %d records" % count


def birmingham(verbose=True, chunksize=50000, limit=None):
    obj = models.Birmingham()
    obj.rewrite_table()
    srid = models.SRID['uk']

    in_file = os.path.join(settings.DATA_DIR, 'birmingham', 'data_090301_140831_matched.csv')
    count = 1
    with open(in_file, 'r') as f:
        c = csv.DictReader(f)
        res = []
        for row in c:
            if limit and count > limit:
                break
            t = {
                'crime_number': sql_quote(row['CRIME_NO']),
                'offence': sql_quote(row['OFFENCE']),
                'datetime_start': "to_timestamp('{} {:02d}{:02d}', 'DD/MM/YYYY HH24MI')".format(
                    row['DATE_START'],
                    int(row['HR_START']),
                    int(row['MIN_START']),
                ),
                'datetime_end': "to_timestamp('{} {:02d}{:02d}', 'DD/MM/YYYY HH24MI')".format(
                    row['DATE_END'],
                    int(row['HR_END']),
                    int(row['MIN_END']),
                ),
                'location': "ST_SetSRID(ST_Point({0}, {1}), {2})".format(
                    row['EASTING'],
                    row['NORTHING'],
                    srid
                ),
                'address': sql_quote(row['FULL_LOC'].replace("'", "")),
                'officer_statement': sql_quote(re.sub(r'[^\x00-\x7f]', r'', row['MO']).replace("'", ""))
            }
            res.append(t)
            count += 1
            if (count % chunksize) == 0 and count:
                obj.insert_many(res)
                res = []
    obj.insert_many(res)
    if verbose:
        print "Inserted %d records" % count
