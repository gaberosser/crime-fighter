__author__ = 'gabriel'
import models
from django.contrib.gis.geos import GEOSGeometry
from django.db import connection
from django.db.models import Q, Count, Sum, Min, Max
import datetime
import pytz
import collections
import numpy as np
UK_TZ = pytz.timezone('Europe/London')

CAD_GEO_CUTOFF = datetime.datetime(2011, 8, 1, 0, tzinfo=UK_TZ)

def month_iterator(start_date, end_date):
    try:
        start_date.tzinfo
    except AttributeError:
        kwargs = {}
    else:
        kwargs = {'tzinfo': pytz.utc}

    this_type = type(start_date)
    sd = start_date
    while sd < end_date:
        next_year = sd.year
        next_month = sd.month + 1
        if sd.month == 12:
            next_year += 1
            next_month = 1
        ed = this_type(next_year, next_month, 1, **kwargs)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


def week_iterator(start_date, end_date):

    sd = start_date
    while sd < end_date:
        ed = sd + datetime.timedelta(days=7)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


def n_day_iterator(start_date, end_date, days=1):
    sd = start_date
    while sd < end_date:
        ed = sd + datetime.timedelta(days=days)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


def fetch_cad_data(nicl_type=None, dedupe=True, only_new=False):

    """
    Fetch relevant data from the Cad table.  NB current uncertainty as to how to process results that are initially
    graded differently to the requested type, but later regraded.  This current query does NOT return these.  The manual
    Python search code DOES return these, but on the date that they are reclassified, which is also incorrect!
    :param nicl_type: integer or iterable of integers corresponding to crime types of interest
    :param dedupe: boolean, True indicates that entries should be unique by CRIS ID
    :param only_new: returns only those results that are NOT snapped to a 250m grid square
    :return: deferred queryset containing desired records.
    """
    ## TODO: test me
    sql = ["""SELECT d.inc_datetime, ST_AsText(d.att_map) pt, d.cl01_id, d.cl02_id, d.cl03_id, d.id FROM database_cad d"""]
    if dedupe:
      sql.append("""JOIN (SELECT cris_entry, MIN(inc_datetime) md, MIN(id) mid FROM database_cad GROUP BY cris_entry) e
      ON d.cris_entry = e.cris_entry AND d.id = e.mid""")
    sql.append("""WHERE NOT (d.cris_entry ISNULL OR d.cris_entry = 'NOT CRIMED')""")
    sql.append("""AND NOT d.att_map ISNULL""")
    if nicl_type:
        try:
            n = tuple(nicl_type)
        except TypeError:
            n = (nicl_type, )
        nicl_str = "(%s)" % ",".join([str(x) for x in n])
        sql.append("""AND (d.cl01_id IN {0} OR d.cl02_id IN {0} OR d.cl03_id IN {0})""".format(nicl_str))
    if only_new:
        sql.append("""AND d.inc_datetime >= '%s'""" % CAD_GEO_CUTOFF.strftime('%Y-%m-%d'))
    sql.append("""ORDER BY d.inc_datetime""")
    # print("\n".join(sql))
    return models.Cad.objects.raw(" ".join(sql))


def fetch_buffered_camden(buf=None):
    """
    Return the (multi)polygon corresponding to Camden, buffered by the specified amount.
    If no buffer is provided, choose one sufficiently large to include ALL Cad entries
    :param buf: Buffer radius in metres
    :return: GEOS Multipolygon
    """
    cursor = connection.cursor()
    if not buf:
        sql = """SELECT 1.05*MAX(ST_Distance(att_map,
        (SELECT mpoly FROM database_division WHERE name='Camden' AND type_id='borough')
        )) FROM database_cad WHERE NOT att_map ISNULL"""
        cursor.execute(sql)
        buf = cursor.fetchone()[0]

    sql = """SELECT ST_AsGeoJson(ST_Buffer(mpoly, %s)) FROM database_division WHERE name='Camden' AND type_id='borough'"""
    cursor.execute(sql, (buf, ))
    return GEOSGeometry(cursor.fetchone()[0])


def initial_filter_cad_old(nicl_type=None, only_new=False):
    """ Extract CAD data with CRIS entry and geolocation.  Returns QuerySet. """
    qset = models.Cad.objects.exclude(cris_entry__isnull=True).exclude(cris_entry__startswith='NOT')\
        .exclude(att_map__isnull=True)
    if nicl_type:
        qry = Q(cl01=nicl_type) | Q(cl02=nicl_type) | Q(cl03=nicl_type)
        qset = qset.filter(qry)
    if only_new:
        return qset.filter(inc_datetime__gte=CAD_GEO_CUTOFF)
    else:
        return qset


def initial_filter_cad(nicl_type=None, only_new=False, spatial_domain=None):
    """ Extract CAD data with CRIS entry and geolocation.  Returns QuerySet. """
    exclude_qry = collections.OrderedDict(
        {
            'cris_entry__isnull': True,
            'cris_entry__startswith': 'NOT',
            'att_map__isnull': True
        }
    )
    filter_qry = collections.OrderedDict()
    if only_new:
        filter_qry['inc_datetime__gte'] = CAD_GEO_CUTOFF

    if spatial_domain:
        filter_qry['att_map__intersects'] = spatial_domain

    qset = models.Cad.objects
    for k, v in exclude_qry.items():
        qset = qset.exclude(**{k: v})

    qset = qset.filter(**filter_qry)

    if nicl_type:
        qry = Q(cl01=nicl_type) | Q(cl02=nicl_type) | Q(cl03=nicl_type)
        qset = qset.filter(qry)

    return qset


def dedupe_cad(qset):
    d = collections.defaultdict(list)
    # de-dupe by CRIS entry
    for q in qset:
        d[q.cris_entry].append(q)
    d = collections.OrderedDict([x for x in sorted(d.items())])
    # take first reported incident date in the case of duplicates
    deduped = [y[np.argmin([x.inc_datetime for x in y])] for y in d.values()]
    return deduped


def clean_dedupe_cad(nicl_type=None, only_new=False, spatial_domain=None):
    """ Extract deduped CAD data with CRIS entry and geolocation.  Returns list. """
    qset = initial_filter_cad(nicl_type=nicl_type, only_new=only_new, spatial_domain=spatial_domain)
    return dedupe_cad(qset)


def cad_queryset_to_r(qset, outfile='from_python.gzip'):
    from rpy2.robjects import r
    import pandas.rpy.common as com
    from pandas import DataFrame
    rel_dt = np.min([x.inc_datetime for x in qset])
    res = np.array([[(x.inc_datetime - rel_dt).total_seconds()] + list(x.att_map.coords) for x in qset])
    df = com.convert_to_r_dataframe(DataFrame(res))
    r.assign("foo", df)
    r("save(foo, file='%s', compress=TRUE)" % outfile)


def time_aggregate_data(cad_list, bucket_dict=None):
        bucket_dict = bucket_dict = bucket_dict or {'all': lambda x: True}
        res = collections.OrderedDict()
        for k, func in bucket_dict.items():
            res[k] = [x for x in cad_list if func(x)]

        return res


def cad_aggregate_daily(cad_list):
    start_date = min([x.inc_datetime for x in cad_list]).replace(hour=0, minute=0, second=0)
    end_date = max([x.inc_datetime for x in cad_list])
    gen_day = n_day_iterator(start_date, end_date)
    res = collections.defaultdict(list)
    for sd, ed in gen_day:
        res[sd].extend([x for x in cad_list if sd <= x.inc_datetime < ed])
    return collections.OrderedDict([x for x in sorted(res.items())])


def cad_aggregate_grid(cad_qset, grid=None, dedupe=True):
    # get grid
    grid = grid or models.Division.objects.filter(type='cad_250m_grid')
    res = collections.OrderedDict()
    for g in grid:
        if dedupe:
            res[g] = dedupe_cad(cad_qset.filter(att_map__within=g.mpoly))
        else:
            res[g] = list(cad_qset.filter(att_map__within=g.mpoly))
    return res


def combine_aggregations_into_count(by_time, by_space):
    nrows = len(by_time)
    ncols = len(by_space)
    vt = [[t.id for t in x] for x in by_time.values()]
    vs = [[t.id for t in x] for x in by_space.values()]
    res = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            t = vt[i]
            s = vs[j]
            res[i, j] = np.sum(np.in1d(s, t))

    return res
