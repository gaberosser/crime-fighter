__author__ = 'gabriel'
import models
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


def initial_filter_cad(nicl_type=None, only_new=False):
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


def dedupe_cad(qset):
    d = collections.defaultdict(list)
    # de-dupe by CRIS entry
    for q in qset:
        d[q.cris_entry].append(q)
    d = collections.OrderedDict([x for x in sorted(d.items())])
    # take first reported incident date in the case of duplicates
    deduped = [y[np.argmin([x.inc_datetime for x in y])] for y in d.values()]
    return deduped


def clean_dedupe_cad(nicl_type=None, only_new=False):
    """ Extract deduped CAD data with CRIS entry and geolocation.  Returns list. """
    qset = initial_filter_cad(nicl_type=nicl_type, only_new=only_new)
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
